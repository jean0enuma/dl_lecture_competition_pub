# DL基礎講座2024　最終課題「Visual Question Answering（VQA）」


## 概要
### 最終課題内容：3つのタスクから1つ選び，高い性能となるモデルを開発してください（コンペティション形式）
3つのタスクはそれぞれ以下の通りです．必ず**1つ**を選んで提出してください．
- 脳波分類（[`MEG-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/MEG-competition)ブランチ）: 被験者が画像を見ているときの脳波から，その画像がどのクラスに属するかを分類する．
  - サンプル数: 訓練65,728サンプル，検証16,432サンプル，テスト16,432サンプル．
  - 入力: 脳波データ．
  - 出力: 1854クラスのラベル．
  - 評価指標: top-10 accuracy（モデルの予測確率トップ10に正解クラスが含まれているかどうか）．
- Visual Question Answering（VQA）（[`VQA-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/VQA-competition)ブランチ）: 画像と質問から，回答を予測する．
  - サンプル数: 訓練19,873サンプル，テスト4,969サンプル．
  - 入力: 画像データ（RGB，サイズは画像によって異なる），質問文（サンプルごとに長さは異なる）．
  - 出力: 回答文（サンプルごとに長さは異なる）．
  - 評価指標: VQAでの評価指標（[こちら](https://visualqa.org/evaluation.html)を参照）を利用．
- Optical Flow Prediction from Event Camera (EventCamera)（[`event-camera-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/event-camera-competition)ブランチ）: イベントカメラのデータから，Optical Flowを予測する．
  - サンプル数: 訓練7,800サンプル，テスト2,100サンプル．
  - 入力: イベントデータ（各時刻，どのピクセルで"log intensity"に変化があったかを記録）．
  - 出力: Optical Flow（連続フレーム間で，各ピクセルの動きをベクトルで表したもの）．
  - 評価指標: Average Endpoint Error（推定したOptical Flowと正解のOptical Flowのユークリッド距離）．

## 環境構築
### Conda
```bash
conda create -n dl_competition python=3.10
pip install -r requirements.txt
```
### Docker
- Dockerイメージのcudaバージョンについては，ご自身が利用するGPUに合わせて変更してください．
```bash

docker build -t <イメージ名> .
docker run -it -v $PWD:/workspace -w /workspace <イメージ名> bash
=======
$ cd dl_lecture_competition_pub
$ git checkout [Competition name]
```
4. README.mdの`環境構築`を参考に環境を作成します．
- README.mdにはconda，もしくはDockerを利用した環境構築の手順を記載しています．
5. README.mdの`ベースラインモデルを動かす`を参考に，ベースラインコードを実行すると，学習や予測が実行され，テストデータに対する予測である`submission.npy`が出力されます．

### 取り組み方
- ベースラインコードを書き換える形で，より性能の高いモデルの作成を目指してください．
  -  基本的には`main.py`などを書き換える形で実装してください．
  -  自分で1から実装しても構いませんが，ベースラインコードと同じ訓練データおよびテストデータを利用し，同じ形式で予測結果を出力してください．
- コンペティションでは，受講生の皆様に`main.py`の中身を書き換えて，より性能の高いモデルを作成していただき，予測結果(`submission.npy`)，工夫点レポート(`.pdf`)，実装したコードのリポジトリのURLを提出いただきます．
- 以下の条件を全て満たした場合に，最終課題提出と認めます．
   - 全ての提出物が提出されていること．
     - 注意：Omicampusで提出した結果が，レポートで書いた内容やコードと大きく異なる場合は，提出と認めない場合があります．
   - Omnicampusでの採点で各タスクのベースライン実装の性能を超えていること．
     - ベースライン性能は各タスクのブランチのREADMEを確認して下さい． 

### Githubへのpush方法
最終課題ではforkしたリポジトリをpublicに設定していただき，皆様のコードを評価に利用いたします．そのため，作成したコードをgithubへpushしていただく必要があります．

以下にgithubにその変更を反映するためのpushの方法を記載します．
1. `git add`
- 以下のように，`git add`に続けて変更を加えたファイル名を空白を入れて羅列します．
```bash
$ git add main.py hogehoge.txt hugahuga.txt
```
2. `git commit`
- `-m`オプションによりメモを残すことができます．その時の変更によって何をしたか，この後何をするのかなど記録しておくと便利です．
```bash
$ git commit -m "hogehoge"
``` 
3. `git push`
- branch nameには提出方法の手順3でcheckoutの後ろで指定したブランチ名を入力します．
```bash
$ git push origin [branch name]
>>>>>>> b22a9bb1792e8dde72521ce42585cf7efff985bb
```

## ベースラインモデルを動かす
### データのダウンロード
- [こちら](https://drive.google.com/drive/folders/1QTcWMATZ_iGsHnxq6-3aXa7D5VZAzs5T?usp=sharing)から各データをダウンロードしてください．
  - train.json: 訓練データのjsonファイル．画像のパス，質問文，回答文がjson形式でまとめられている．
  - valid.json: テストデータのjsonファイル．画像のパス，質問文がjson形式でまとめられている．
  - train.zip: 訓練データの画像ファイル．
  - valid.zip: テストデータの画像ファイル．
- ダウンロード後，train.zipとvalid.zipを解凍し，各データをdataディレクトリ下に置いてください．
### 訓練・提出ファイル作成
```bash
python3 main.py
```
- `main.py`と同様のディレクトリ内に，学習したモデルのパラメータ`model.pth`とテストデータに対する予測結果`submission.npy`ファイルが作成されます．
- ベースラインは非常に単純な手法のため，改善の余地が多くあります．VQAでは**Omnicampusにおいてベースラインのtest accuracy=41%を超えた提出のみ，修了要件として認めることとします．**

## VizWiz(2023 edition) dataset [[link](https://www.kaggle.com/datasets/nqa112/vizwiz-2023-edition)] の詳細
- 24842枚の画像があり，訓練データには各画像に対して1つの質問と10人の回答者による回答．
  - 10人の回答はすべて同じであるとは限らない．
- 24842のデータの内，80%(19873)が訓練データ，20%(4969)がテストデータとして与えられる．
  - テストデータに対する回答は正解ラベルとし，訓練時には与えられない．
  - データ提供元ではtrainとvalに分かれているが，データの配布前に運営の方でtrainとvalをランダムに分け直している．

## タスクの詳細
- 本コンペでは，与えられた画像と質問に対し，適切な回答をモデルに出力させる．
- 評価は[VQA](https://visualqa.org/index.html)(Visual Question Answering)に基づき，以下の式で計算される．
$$\text{Acc}(ans) = min(\frac{humans \ that \ said \ ans}{3}, 1)$$
- 1つのデータに対し，10人の回答の内9人の回答を選択し上記の式で性能を評価した，10パターンのAccの平均をそのデータに対するAccとする．
- 予測結果と正解ラベルを比較する前に，回答をlowercaseにする，冠詞は削除するなどの前処理を行う（[詳細](https://visualqa.org/evaluation.html)）．

## 考えられる工夫の例
- 質問文の前処理．
  - 回答については評価指標に合わせるため前処理をしていますが，質問文には前処理を施していません．そのため同じ単語であっても大文字，小文字で異なる単語として扱われます．大文字・小文字の統一，冠詞の削除など適切な処理をすることで，性能向上が見込めます．
- 質問文の表現．
  - ベースラインでは，質問文をモデルに入力する際にはone-hotベクトルにしています．この表現をtokenizer等を利用して分散表現にすることで，モデルが学習しやすい表現となり，性能向上が見込めます．
- 回答の出力候補の変更．
  - ベースラインは非常に簡素なモデルとなっており，出力は訓練データに存在する回答例の中からどれが適切かを選択する，クラス分類の形式になっています．これでは訓練データには存在しない回答は出力不可能なため，より大きなコーパスを用いると良いでしょう（huggingfaceでは，VizWizに利用できる[class_mapping](https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv)が存在します）．
- 画像の前処理．
  - 画像の前処理には形状を同じにするためのResizeのみを利用しています．第5回の演習で紹介したようなデータ拡張を追加することで，疑似的にデータを増やし汎化性能の向上が見込めます．
