# Description
簡単に3Dモデルに生まれ変われるアプリ「ReBorn」のフレームワーク部分

# Requirement
- ffmpeg
- 

# DB
```
room_name | model_path | background_path | sound_path | vmd_path |  subtitle_path
```
- プライマリキーは`room_name`
    - その関係上、`room_name`は日本語を含まないほうが望ましい
- すべてString
- `app.py`の`Entry`クラスで定義しているのでそちらも参照

# System
## ルート
- ルームを作るページ
- room_nameなどを入力
- room_nameが使用可能か判定
- DBに保存
- Vstudio(?room_name=hogehoge)に遷移
- (できれば)ajaxでモデルをリアルタイム表示
- (できれば)編集パスワードもDBに保存

## Vstudio(?room_name)
- 録画開始ボタンを表示
- jsで録画＆録音
- 録画中はタイマーを表示
- 終わったらPythonで処理
- それか直接vmdを貼り付け
- vmdを得たらDBに保存

## Vroom(?room_name)
- DBからroom_nameに対応する各種データを取り出し
- それを使い, Three.jsでroomを表示

# todo
- カメラ＆マイクからmp4出力 -> Done!!
- mp4からwav抽出 -> Done!!
    - blobから直接変換するのはライブラリの仕様的に無理っぽい
- 背景 -> Done!!
- 変換後音声(ダミー)をdbに登録して、roomで表示できるように
    - app.py->vmdmake: 
        - 変換後の音声を特定のディレクトリに保存して、ダミーの音声を返す関数
        - dbにアクセスして、変換後の音声pathを保存 
    - Vstudio.html: 
        - vmd選択を外す(コメントアウト)
    - Vroom.html:
        - BGMの他、音声(voice)も受け取る
        - voiceも同時にsceneにadd(両方無理そうなら)
        - できれば音声と動画に再生
- mmd選択画面のajax 
- qrコード
- 編集画面のパスコード
- vmd直接アップロード
- 動画アップロード
- デプロイ