import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm

# 学習済みモデル
from torchvision.models import resnet18, resnet50, vit_b_16
from transformers import BertTokenizer
import math
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = self.tokenizer.tokenize(question)
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.question2idx["[UNK]"] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    words = answer["answer"]
                    words = process_text(words)
                    if words not in self.answer2idx:
                        self.answer2idx[words] = len(self.answer2idx)
            self.answer2idx["[UNK]"] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def max_len(self):
        """
        質問文の最大の長さを返す．
        """
        max_len = 0
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            max_len = max(max_len, len(words))
        return max_len

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question_words = process_text(self.df["question"][idx])
        question_words = self.tokenizer.tokenize(question_words)
        question = []
        for word in question_words:
            try:
                question.append(self.question2idx[word])
            except KeyError:
                question.append(self.question2idx["[UNK]"])
        question_length = len(question)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            confidences = [answer["answer_confidence"] for answer in self.df["answers"][idx]]
            # confidenceがyesなら2,maybeなら1,noなら0に変換
            confidences = [2 if confidence == "yes" else 1 if confidence == "maybe" else 0 for confidence in
                           confidences]
            confidences=mode(confidences)
            # confidenceが
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            return image, torch.Tensor(question), torch.Tensor(answers), confidences, mode_answer_idx, question_length

        else:
            return image, torch.Tensor(question), question_length

    def __len__(self):
        return len(self.df)

    @staticmethod
    def collate_fn(batch):
        """
        バッチサイズ分のデータをまとめる．

        """
        images = []
        questions = []
        answers = []
        mode_answer_idxs = []
        question_lengths = []
        confidences = []
        if len(batch[0]) == 6:
            for image, question, answer, confidence, mode_answer_idx, question_length in batch:
                images.append(image)
                questions.append(question)
                answers.append(answer)
                mode_answer_idxs.append(mode_answer_idx)
                question_lengths.append(question_length)
                confidences.append(confidence)
            confidences = torch.Tensor(confidences).long()
            mode_answer_idxs = torch.Tensor(mode_answer_idxs).int()
            question_lengths = torch.Tensor(question_lengths).int()
            images = torch.stack(images, dim=0)
            answers = torch.stack(answers, dim=0)
            questions_batch = torch.zeros(len(questions), int(max(question_lengths)), dtype=torch.long)
            for i, question in enumerate(questions):
                questions_batch[i, :len(question)] = question

            return images, questions_batch, answers, confidences, mode_answer_idxs, question_lengths
        else:
            for image, question, question_length in batch:
                images.append(image)
                questions.append(question)
                question_lengths.append(question_length)

            images = torch.stack(images, dim=0)
            question_lengths = torch.Tensor(question_lengths)
            questions_batch = torch.zeros(len(questions), int(max(question_lengths)), dtype=torch.long)
            for i, question in enumerate(questions):
                questions_batch[i, :len(question)] = question

            return images, questions_batch, question_lengths


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layers, num_encoder_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src, src_length, feature=None):
        src = self.embedding(src)
        if feature is not None:
            src = torch.cat([feature, src], dim=1)
            src_length = src_length + feature.size(1)
        src = torch.cat([self.cls.expand(src.size(0), -1, -1), src], dim=1)
        src = src + self.pe(src)
        mask = self.create_padding_mask(src, src_length + 1).to(src.device)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        return output[:, 0]

    def create_padding_mask(self, src, src_length):
        mask = (torch.arange(src.size(1)))[None, :] >= src_length[:, None]
        return mask


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int, num_features: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        # self.resnet = resnet18(weights='DEFAULT')
        self.vit = vit_b_16()
        self.vit.heads = nn.Identity()
        self.text_encoder = TransformerEncoder(vocab_size, num_features, nhead, num_encoder_layers, dim_feedforward,
                                               dropout)

        self.fc = nn.Sequential(
            nn.Linear(num_features*2, num_features * 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 4, n_answer)
        )
        self.fc_type = nn.Sequential(
            nn.Linear(num_features*2, num_features * 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 4, 3))

    def forward(self, image, question, question_length, is_clip=False):
        image_feature = self.vit(image)  # 画像の特徴量
        batch_size, vector_size = image_feature.size(0), image_feature.size(1)
        text_feature = self.text_encoder(question, question_length)  # テキストの特徴量
        # text_feature: [batch_size, vector_size]
        if is_clip:
            assert text_feature.size(1) == image_feature.size(1)
            image_feature = F.normalize(image_feature, p=2, dim=1)
            text_feature = F.normalize(text_feature, p=2, dim=1)
            feature = torch.bmm(image_feature.unsqueeze(2), text_feature.unsqueeze(1)) * 100
            return feature
        else:
            feature = torch.cat([image_feature, text_feature], dim=1)
            x = self.fc(feature)
            x_type = self.fc_type(feature)
            return x, x_type


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device, batch_size=64, is_clip=False):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0
    scaler = torch.cuda.amp.GradScaler(4096)
    start = time.time()
    for i, (image, question, answers, confidences, mode_answer, question_length) in tqdm(enumerate(dataloader),
                                                                                         total=len(
                                                                                             dataloader.dataset) // batch_size):
        image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(
            device), mode_answer.long().to(device)
        with torch.cuda.amp.autocast():
            pred = model(image, question, question_length, is_clip=is_clip)
            if is_clip:
                labels = torch.arange(0, pred.size(1)).to(device)
                # pred: [batch_size, n,n]
                batch_size, num_features, _ = pred.size()
                loss1 = criterion(pred.reshape(pred.size(0) * pred.size(1), -1),
                                  labels.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1))
                loss2 = criterion(pred.transpose(1, 2).reshape(pred.size(0) * pred.size(1), -1),
                                  labels.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1))
                loss = (loss1 + loss2) / 2
            else:
                loss = criterion(pred[0], mode_answer.squeeze())
                loss+=criterion(pred[1], confidences.squeeze().to(device))/2

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # クリップ時に正しくできるように一度スケールを戻す
        scaler.unscale_(optimizer)
        # 大きすぎる勾配をクリップ
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if not is_clip:
            total_acc += VQA_criterion(pred[0].argmax(1), answers)  # VQA accuracy
            simple_acc += (pred[0].argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        else:
            total_acc += 0
            simple_acc += 0

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()

    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        with torch.cuda.amp.autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=test_transform,
                              answer=False)
    test_dataset.update_dict(train_dataset)
    print(f"train dataset: {len(train_dataset)}")
    print(f"test dataset: {len(test_dataset)}")
    print(f"question vocab size: {len(train_dataset.question2idx)}")
    print(f"answer vocab size: {len(train_dataset.answer2idx)}")
    print(f"question max length: {train_dataset.max_len()}")

    batch_size = 32



    # optimizer / criterion
    num_epoch = 100
    model = VQAModel(vocab_size=len(train_dataset.question2idx) + 1, n_answer=len(train_dataset.answer2idx),
                     num_features=768, nhead=8, num_encoder_layers=12, dim_feedforward=1024,
                     dropout=0.5).float().to(
        device)
    is_clips=[True,False]
    for is_clip in is_clips:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=8, drop_last=True, collate_fn=train_dataset.collate_fn)



        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # train model
        print("start pretraining...")
        for epoch in range(num_epoch):
            train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device,
                                                                        batch_size,is_clip=is_clip)
            print(f"【{epoch + 1}/{num_epoch}】\n"
                  f"train time: {train_time:.2f} [s]\n"
                  f"train loss: {train_loss:.4f}\n"
                  f"train acc: {train_acc:.4f}\n"
                  f"train simple acc: {train_simple_acc:.4f}")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "is_clip":is_clip
            }, f"./checkpoint/checkpoint.cpt")
    # 提出用ファイルの作成
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                              num_workers=8,
                                              drop_last=False, collate_fn=train_dataset.collate_fn)
    submission = []
    for image, question, question_length in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question, question_length)
        pred = pred[0].argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', True)
    main()
