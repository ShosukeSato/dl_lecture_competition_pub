# 必要なライブラリのインポート
import re  # 正規表現処理用
import random  # 乱数生成用
import time  # 時間計測用
from statistics import mode  # 最頻値計算用
from PIL import Image  # 画像処理用
import numpy as np  # 数値計算用
import pandas  # データフレーム処理用
import torch  # PyTorch深層学習フレームワーク
import torch.nn as nn  # PyTorchのニューラルネットワークモジュール
import torchvision  # 画像処理用PyTorchライブラリ
from torchvision import transforms  # 画像変換用

# 乱数のシードを設定する関数
def set_seed(seed):
    """
    再現性を確保するために、各種乱数生成器のシードを設定する
    
    Args:
        seed (int): 設定するシード値
    """
    random.seed(seed)  # Pythonの組み込み乱数生成器のシード設定
    np.random.seed(seed)  # NumPyの乱数生成器のシード設定
    torch.manual_seed(seed)  # PyTorchのCPU乱数生成器のシード設定
    torch.cuda.manual_seed(seed)  # PyTorchのGPU乱数生成器のシード設定（単一GPU）
    torch.cuda.manual_seed_all(seed)  # PyTorchのGPU乱数生成器のシード設定（複数GPU）
    torch.backends.cudnn.deterministic = True  # CuDNNを決定論的モードに設定
    torch.backends.cudnn.benchmark = False  # CuDNNのベンチマーク機能を無効化

# テキストを前処理する関数
def process_text(text):
    """
    入力テキストを前処理する
    
    Args:
        text (str): 前処理する入力テキスト
    
    Returns:
        str: 前処理されたテキスト
    """
    # すべての文字を小文字に変換
    text = text.lower()
    
    # 数詞を数字に変換するための辞書
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    # 数詞を数字に置換
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    
    # 単独のピリオド（小数点でないもの）を削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    
    # 冠詞（a, an, the）を削除
    text = re.sub(r'\b(a|an|the)\b', '', text)
    
    # 一般的な短縮形にアポストロフィを追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    
    # 句読点をスペースに変換（アポストロフィとコロンは除く）
    text = re.sub(r"[^\w\s':]", ' ', text)
    
    # カンマの前のスペースを削除
    text = re.sub(r'\s+,', ',', text)
    
    # 連続するスペースを1つに変換し、前後の空白を削除
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        # コンストラクタ：データセットの初期化
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer  # 回答を含むかどうかのフラグ

        # 質問と回答のインデックス辞書を初期化
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            processed_question = process_text(question)  # テキストの前処理
            words = processed_question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    # 新しい単語にインデックスを割り当て
                    self.question2idx[word] = len(self.question2idx)

        # 前処理ずみの質問文を新しい列として追加
        self.df["processed_question"] = self.df["question"].apply(process_text)
        
        # 質問のインデックスから単語への逆変換辞書を作成
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加（回答がある場合のみ）
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)  # テキストの前処理
                    if word not in self.answer2idx:
                        # 新しい回答単語にインデックスを割り当て
                        self.answer2idx[word] = len(self.answer2idx)
            
            # 回答のインデックスから単語への逆変換辞書を作成
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．
        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        # 訓練データセットの辞書で、現在のデータセットの辞書を更新
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
        # 画像の読み込みと前処理
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        # # 質問のone-hot表現の生成
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # question_words = self.df["processed_question"][idx].split(" ")
        # for word in question_words:
        #     try:
        #         question[self.question2idx[word]] = 1  # one-hot表現に変換
        #     except KeyError:
        #         question[-1] = 1  # 未知語

        # 質問文のインデックス化
        question_words = self.df["processed_question"][idx].split()
        question_indices = [self.question2idx.get(word, len(self.question2idx)) for word in question_words]

        # パディング（最大長を100とする例）
        max_length = 100
        if len(question_indices) > max_length:
            question_indices = question_indices[:max_length]
        else:
            question_indices += [0] * (max_length - len(question_indices))
    
        question_tensor = torch.LongTensor(question_indices)

        if self.answer:
            # 回答の処理（回答がある場合のみ）
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            return image, question_tensor, torch.Tensor(answers), int(mode_answer_idx)
        else:
            # 回答がない場合（テストデータなど）
            return image, question_tensor

    def __len__(self):
        # データセットの長さ（サンプル数）を返す
        return len(self.df)
    
# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    """
    VQAタスクの評価指標を計算する関数

    Args:
        batch_pred (torch.Tensor): モデルの予測結果のバッチ
        batch_answers (torch.Tensor): 正解の回答のバッチ（各質問に対して複数の回答を含む）

    Returns:
        float: バッチ全体の平均精度スコア
    """
    total_acc = 0.  # バッチ全体の累積精度

    # バッチ内の各予測と回答セットに対してループ
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.  # 現在の質問に対する精度
        
        # 各回答に対してループ
        for i in range(len(answers)):
            num_match = 0  # 予測と一致する他の回答の数
            
            # 他の全ての回答と比較
            for j in range(len(answers)):
                if i == j:
                    continue  # 同じ回答はスキップ
                if pred == answers[j]:
                    num_match += 1  # 予測が他の回答と一致した場合、カウントを増やす
            
            # 精度にスコアを追加（最大1まで）
            acc += min(num_match / 3, 1)
        
        # 質問ごとの平均精度を累積精度に追加
        total_acc += acc / 10  # 10は回答の総数（仮定）

    # バッチ全体の平均精度を返す
    return total_acc / len(batch_pred)

# 3. モデルの実装
# ResNetを利用できるようにしておく

class BasicBlock(nn.Module):
    """ResNetの基本ブロック"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        BasicBlockの初期化
        Args:
            in_channels (int): 入力チャンネル数
            out_channels (int): 出力チャンネル数
            stride (int): ストライド（デフォルト: 1）
        """
        super().__init__()

        # 1つ目の畳み込み層とバッチ正規化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 2つ目の畳み込み層とバッチ正規化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続（ショートカット）の定義
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """順伝播"""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)  # スキップ接続の追加
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """ResNetのボトルネックブロック"""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        BottleneckBlockの初期化
        Args:
            in_channels (int): 入力チャンネル数
            out_channels (int): 出力チャンネル数
            stride (int): ストライド（デフォルト: 1）
        """
        super().__init__()

        # 1x1の畳み込み層でチャンネル数を減らす
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3の畳み込み層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1の畳み込み層でチャンネル数を増やす
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続（ショートカット）の定義
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        """順伝播"""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)  # スキップ接続の追加
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNetモデル"""
    def __init__(self, block, layers):
        """
        ResNetの初期化
        Args:
            block: 使用するブロック（BasicBlockまたはBottleneckBlock）
            layers (list): 各層のブロック数
        """
        super().__init__()
        self.in_channels = 64

        # 入力層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetの主要な層
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        # 出力層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        """ResNetの層を作成するヘルパーメソッド"""
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """順伝播"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    """ResNet18モデルを作成"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    """ResNet50モデルを作成"""
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    """Visual Question Answering (VQA) モデル"""
    def __init__(self, vocab_size: int, embed_size: int, n_answer: int):
        """
        VQAモデルの初期化
        Args:
            vocab_size (int): 質問の語彙サイズ
            n_answer (int): 可能な回答の数
        """
        super().__init__()
        self.resnet = ResNet18()  # 画像特徴抽出器としてResNet18を使用
        # self.text_encoder = nn.Linear(vocab_size, 512)  # テキスト特徴抽出器
        self.embedding = nn.Embedding(vocab_size + 1, embed_size)  # +1 for unknown words
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_size, 512),
            nn.ReLU(inplace=True)
        )

        # 特徴量を結合し、最終的な回答を予測する全結合層
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        """
        順伝播
        Args:
            image: 入力画像
            question: 入力質問（one-hot encoded）
        Returns:
            回答の確率分布
        """
        image_feature = self.resnet(image)  # 画像の特徴量
        # 質問文の埋め込みと特徴抽出
        embedded = self.embedding(question)
    
        # パディングを無視して平均を計算
        mask = (question != 0).float().unsqueeze(-1)
        question_feature = (embedded * mask).sum(dim=1) / mask.sum(dim=1)
        question_feature = self.text_encoder(question_feature)

        x = torch.cat([image_feature, question_feature], dim=1)  # 特徴量の結合
        x = self.fc(x)  # 全結合層を通して回答を予測

        return x
    
# 4. 学習の実装

def train(model, dataloader, optimizer, criterion, device):
    """
    モデルの訓練を行う関数
    
    Args:
        model: 訓練するモデル
        dataloader: 訓練データのDataLoader
        optimizer: 最適化アルゴリズム
        criterion: 損失関数
        device: 使用するデバイス（CPUまたはGPU）
    
    Returns:
        平均損失、VQA精度、単純精度、訓練時間
    """
    model.train()  # モデルを訓練モードに設定
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        # データをデバイスに移動
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        
        pred = model(image, question)  # モデルによる予測
        loss = criterion(pred, mode_answer.squeeze())  # 損失の計算
        
        optimizer.zero_grad()  # 勾配をゼロに初期化
        loss.backward()  # 勾配の計算
        optimizer.step()  # パラメータの更新
        
        total_loss += loss.item()  # 累積損失の更新
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    # 平均損失、精度、訓練時間を計算して返す
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    """
    モデルの評価を行う関数
    
    Args:
        model: 評価するモデル
        dataloader: 評価データのDataLoader
        optimizer: 最適化アルゴリズム（この関数では使用しない）
        criterion: 損失関数
        device: 使用するデバイス（CPUまたはGPU）
    
    Returns:
        平均損失、VQA精度、単純精度、評価時間
    """
    model.eval()  # モデルを評価モードに設定
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    with torch.no_grad():  # 勾配計算を無効化
        for image, question, answers, mode_answer in dataloader:
            # データをデバイスに移動
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
            
            pred = model(image, question)  # モデルによる予測
            loss = criterion(pred, mode_answer.squeeze())  # 損失の計算
            
            total_loss += loss.item()  # 累積損失の更新
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    # 平均損失、精度、評価時間を計算して返す
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    # deviceの設定
    set_seed(42)  # 再現性のためにシードを設定
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが利用可能な場合はGPUを使用

    # dataloader / model
    # 画像の前処理を定義
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 画像サイズを224x224にリサイズ
        transforms.ToTensor()  # テンソルに変換
    ])
    
    # 訓練データセットとテストデータセットの作成
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)  # テストデータセットの辞書を訓練データセットに合わせて更新
    
    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # モデルの初期化
    model = VQAModel(
        vocab_size=len(train_dataset.question2idx),
        embed_size=300,  # 埋め込みの次元数を設定
        n_answer=len(train_dataset.answer2idx)
    ).to(device)

    # optimizer / criterion
    num_epoch = 20  # エポック数
    criterion = nn.CrossEntropyLoss()  # 損失関数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 最適化アルゴリズム

    # train model
    for epoch in range(num_epoch):
        # モデルの訓練
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        
        # 訓練結果の表示
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()  # モデルを評価モードに設定
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)  # モデルによる予測
        pred = pred.argmax(1).cpu().item()  # 最も確率の高い回答のインデックスを取得
        submission.append(pred)
    
    # 予測結果をインデックスから実際の回答に変換
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    
    # モデルと予測結果の保存
    torch.save(model.state_dict(), "model.pth")  # モデルの重みを保存
    np.save("submission.npy", submission)  # 予測結果を保存

if __name__ == "__main__":
    main()
