# Libtorch_example_1
Some examples of libtorch (Pytorch for C++ API).

libtorch (Pytorch の c++ API) の使い方がわからなかったので、忘備録として残す。ちなみに僕は機械学習はド素人です。

# 説明
このコードは排他的論理和を正しく推論する全結合ニューラルネットワークを作ることを目的としている。排他的論理和は
```
入力 -> 出力
0, 0 -> 0
0, 1 -> 1
1, 0 -> 1
1, 1 -> 0
```
のように4通りのパターンしかないが、簡単なものを理解すれば大体応用はすぐにできたりするので、これを採用している。
これは２つの入力に対して 0 か 1 かを出力すればよいので、分類の問題に帰着できる。ニューラルネットワークとしては、出力・入力のニューロン数がそれぞれ２つのものを設計する。このコードでは、ニューロン数は入力層から 2:4:4:2 個としている。
これを念頭に置いて、コードを眺めていく。

ニューラルネットワーク設計部分
-
以下がニューラルネットワークを定義する部分。何が書いてあるかは大体わかると思う。
```c++
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 4));
        fc2 = register_module("fc2", torch::nn::Linear(4, 4));
        fc3 = register_module("fc3", torch::nn::Linear(4, 2));
    }
    torch::nn::Linear fc1 = {nullptr};
    torch::nn::Linear fc2 = {nullptr};
    torch::nn::Linear fc3 = {nullptr};

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::softmax(fc3->forward(x), 1);
        return x;
    }
};
```

`torch::nn::Module` を継承して設定していく。
`torch::nn::Linear` の`Linear`は全結合層を意味する(一応bias項も含まれる)。コンストラクタ内部の `resister_module`関数を使って具体的な層数を決めていく。例えば、入力が2, 出力が4の全結合層だったら、`fc1 = resister_module("名前", (2, 4))`などとする。このとき、層間の出力数と入力数は一致させる。 

`torch::Tensor forward(torch::Tensor x)`部分で、どういう風に伝播させていくか(活性化関数など)を決める。ここでは、途中まで活性化関数としてreluを使っている。今回は分類の問題なので、出力における活性化関数はsoftmaxを使う。今回出力層の数は2つだが、第二引数には1を与える(何故かはよくわかってない...)。

データの準備
-
以下の部分。
```c++
    std::vector<float> in = {0, 0, 0, 1, 1, 0, 1, 1};
    std::vector<float> out = {0, 1, 1, 0, 1, 0, 0, 1};
    int n_data = 4;
    auto input = torch::from_blob(in.data(), {n_data, 2});
    auto output = torch::from_blob(out.data(), {n_data, 2});
```
`vector<float>`で１次元配列として用意しておくと楽。今、入力と出力の対応関係は、
```
入力 -> 出力
{0, 0} -> {0, 1}
{0, 1} -> {1, 0}
{1, 0} -> {1, 0}
{1, 1} -> {0, 1}
```
という風に定義しており、１次元配列として順に並べておく。データ数は4通りしかないので4つ。
次の、`torch::from_blob(in.data(), {n_data, 2})` で、libtorchのTensorに変換する。引数には、第１引数に１次元配列の先頭ポインタ、第２引数にTensorの形を与える。第２引数の意味は行列で考えるとわかりやすい。今のように`{n_data, 2}`としておくと、n_data 行 2 列の行列となる(試してないがコンマでさらに区切ると一般的なテンソルにできると思う)。こうすることで、第 i 行が i 番目のデータに対応し、j 列目は入力層の j 番目に対応するようになる。出力層も同様。

学習
-

以下の部分。
```c++
    std::shared_ptr<Net> net = std::make_shared<Net>();

    torch::optim::Adam optimizer(net->parameters());

    for (int i = 0; i < 4000; ++i) {
        auto out_ = net->forward(input);
        optimizer.zero_grad();
        auto loss = torch::binary_cross_entropy(out_, output);
        auto loss_value = loss.item<float>();
        loss.backward();
        optimizer.step();
        if (i % 50 == 0) {
            std::cout << "====== LEARNING INFO ====== " << std::endl;
            std::cout << "Loss : " << loss_value << std::endl;
        }
    }
    std::cout << net->forward(input) << std::endl;
```

` std::shared_ptr<Net> net = std::make_shared<Net>()` で設計したニューラルネットワークを定義する。今回、optimizer は Adam を使っている。Adam の引数によってパラメータも変えることもできる。

for ループ内が実際に学習させる部分。`auto out_ = net->forward(input)` でニューラルネットワークの出力を得る。`optimizer.zero_grad()` はよく意味が分かっていない。名前の意味から察するに勾配が0になるように学習を進める？ということをしているのだろうが、内部的にどういう処理をしているかは理解していない。`auto loss = torch::binary_cross_entropy(out_, output)` で出力結果と正解との相違(損失)を取得。損失関数は分類問題なのでクロスエントロピーを使った。`loss.backward()` は逆誤差伝搬を意味すると思う。 `optimizer.step()` は何かわからないけど、ここで実際にパラメータを変更している？

これでfor loopを抜けたあと、学習されたニューラルネットワークが得られたことになる。`std::cout << net->forward(input) << std::endl`の出力は以下のようになった(実行するたびに変わるけど一例)。

```
 0.0222  0.9778
 0.9995  0.0005
 0.9991  0.0009
 0.0222  0.9778
[ CPUFloatType{4,2} ]
```

CMakeLists.txt
-

```
list(APPEND CMAKE_PREFIX_PATH ~/Torch/libtorch)
```
の部分でlibtorchまでのパスを書いておけば通る、と思う。あとは
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
の要領で実行ファイルができる。

課題
-
学習はできたけど、学習用データとテスト用データを分ける方法がわからない。学習したあとに自分でテストすればいいけど、やりかたがあるかもしれない。