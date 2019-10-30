# Description
簡単に美少女に生まれ変われるアプリ「ReBorn」をビルドするためのリポジトリです。

# Requirement
Dockerfileを参照

なお、ローカルでビルドする場合gpuにアクセス可能である必要があります。

# Usage
## local build
1. このリポジトリをクローン
2. `sudo docker build -t xxx:0.1 .`
3. `sudo docker run -p 5000:5000 --gpus all -it xxx:0.1`
4. dockerコンテナに入れていることを確認し、`/ReBone_v2`に移動
5. `python app.py`を実行
6. ブラウザで localhost:5000 を開く

## Remote build
now in preparation.

# System
## DB
```
room_name | model_path | background_path | sound_path | vmd_path |  subtitle_path | voice_path
```
- プライマリキーは`room_name`
    - その関係上、`room_name`は日本語を含まないほうが望ましい
- すべてString
- `app.py`の`Entry`クラスで定義しているのでそちらも参照


## Pages
### ルート
- ルームを作るページ
- room_nameなどを入力
- room_nameが使用可能か判定
- DBに保存
- Vstudio(?room_name=hogehoge)に遷移

### Vstudio(?room_name)
- 録画開始ボタンを表示
- jsで録画＆録音
- 録画中はタイマーを表示
- 終わったらPythonで処理
- vmdを得たらDBに保存

### Vroom(?room_name)
- DBからroom_nameに対応するモーション・音声等のデータを取り出す
- それらを使い, Three.jsでroomを表示

# todo
- app.py + vmd-lifting ＆ voice converterの結合 -> done!
- nginx+gnicornで動くように
    - docker-compose.ymlの作成
