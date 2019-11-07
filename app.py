from flask import Flask, render_template, request, url_for
import os
from flask_sqlalchemy import SQLAlchemy
import ffmpeg
import time
from moviepy.editor import *
from rebone_VC import VoiceConverter
from pose_est_mod.pem import video2vmd
import librosa
import tensorflow as tf

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__))+'/uploads'
app.config['STATIC_FOLDER'] = os.path.dirname(os.path.abspath(__file__))+'/static'

# ツール

def get_path(type, name):
    if type == "model":
        # pmd or pmxで場合分けする必要がある
        if name == "miku_1":
            return 'static/models/'+name+'/'+name+'.pmd'
        else:
            return 'static/models/'+name+'/'+name+'.pmx'
    elif type == "background":
        return 'static/backgrounds/'+name+'.jpg'
    elif type == "sound":
        return 'static/sounds/'+name+'.mp3'
    elif type == "vmd":
        return 'static/vmds/'+name+'.vmd'
    elif type == "subtitle":
        return 'static/subtitles/'+name+'.json'

def is_valid(room_name):    
    table = Entry.query.all()
    for row in table:
        if row.room_name == room_name:
            return False
    return True

def is_exist(room_name):
    table = Entry.query.all()
    for row in table:
        if row.room_name == room_name:
            return True
    return False

# ルーティング
@app.route('/', methods=['GET','POST'])
def createroom():
    if request.method == 'POST':
        if(is_valid(request.form["room_name"])):
            add_entry(
                room_name = request.form["room_name"], 
                model_path = get_path('model', request.form["model"]),
                background_path = get_path('background', request.form["background"]),
                sound_path = "", vmd_path = "", subtitle_path = "", voice_path = "") 
            next_url = url_for('Vstudio', room_name=request.form["room_name"])
            return render_template('show_link_and_QRcode.html', url=next_url)
        else:
            return render_template('createroom.html', pre_val=request.form, message="このルーム名は既に使われています。")
    else:
        return render_template('createroom.html', pre_val=[], message = "")


@app.route('/Vstudio', methods=['POST', 'GET'])
def Vstudio():
    if is_exist(request.args.get('room_name','')):
        if request.method == 'POST':
            print(get_path('sound', request.form['sound']))
            update_entry(
                room_name = request.args.get('room_name',''), 
                model_path = None,
                background_path = None,
                sound_path = get_path('sound', request.form['sound']), 
                vmd_path = None,
                subtitle_path = get_path('subtitle', 'sample'),
                voice_path = None ) 
            next_url = url_for('Vroom',room_name=request.args.get('room_name'))
            return render_template('show_link_and_QRcode.html', url=next_url)
        else: 
            return render_template('Vstudio.html', room_name=request.args.get('room_name',''))
    else: 
        return render_template('not_found.html', message="ルームが指定されていないか、指定のルームが見つかりません。")
    

@app.route('/Vroom', methods=['POST', 'GET'])
def Vroom():
    if is_exist(request.args.get('room_name')):
        entry = Entry.query.filter(Entry.room_name == request.args.get('room_name') ).first()
        return render_template('Vroom.html', 
            model_path = entry.model_path,
            background_path = entry.background_path,
            sound_path = entry.sound_path,
            vmd_path = entry.vmd_path,
            subtitle_path = entry.subtitle_path,
            voice_path = entry.voice_path )
    else:
        room_list = get_list_from_db()
        return render_template('list.html', room_list=room_list)


@app.route('/makevmd', methods=['POST', 'GET'])
def makevmd():  # todo: できれば名前変えたい(音声変換もするので)
    if request.method == 'POST':
        video_file = request.files['video_blob']
        video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.webm'))
        time.sleep(1)   # 保存処理に時間がかかるので少し待つ

        # パスを設定(このパスだけは下階層のモジュールからも参照するので絶対パス))
        webm_path = app.config['UPLOAD_FOLDER']+'/video.webm'
        mp4_path = app.config['UPLOAD_FOLDER']+'/video.mp4'
        wav_path = app.config['UPLOAD_FOLDER']+'/audio.wav'
        fps30_mp4_path = app.config['UPLOAD_FOLDER']+'/video_30fps.mp4'

        # webm -> mp4 ＆ wavに変換して保存
        (
            ffmpeg
            .input(webm_path)
            .output(mp4_path, vcodec='h264')
            .run(overwrite_output=True)
        )

        (
            ffmpeg
            .input(webm_path)
            .output(wav_path, acodec='pcm_s16le')
            .run(overwrite_output=True)
        )

        # mp4をfps30にして保存
        clip = VideoFileClip(mp4_path)
        clip.write_videofile(fps30_mp4_path, fps=30)


        # todo: ここで変換処理を呼び出す

        ## 音声変換
        ### input: wav_path, output: processed_wav_path
        processed_wav_path = app.config['STATIC_FOLDER']+'/voices/'+request.args.get('room_name','')+'.wav'
        wav, _ = librosa.load(wav_path)
        vc_result = VoiceConverter.convert_voice(wav)
        librosa.output.write_wav(processed_wav_path, vc_result, sr=22050)
        #メモリ解放
        tf.contrib.keras.backend.clear_session()

        ## 動画変換
        ### input: fps30_mp4_path, output: vmd_path
        vmd_path = app.config['STATIC_FOLDER']+'/vmds/'+request.args.get('room_name','')+'.vmd'
        video2vmd(fps30_mp4_path, vmd_path)
        #メモリ解放
        tf.contrib.keras.backend.clear_session()


        # 音声変換処理で返ってきたパス(processed_wav_path)と
        # 動画変換処理で返ってきたパス(vmd_path)をdbに保存
        update_entry(
            room_name = request.args.get('room_name',''),
            model_path = None,
            background_path = None,
            sound_path = None,
            vmd_path = vmd_path,
            subtitle_path = None,
            voice_path = processed_wav_path
        )

    return "vmd and voice are generated!!"

@app.route('/runanime')
def runanime():
    return render_template('runanime.html') 
 

# データベース
#db_uri = os.environ.get('DATABASE_URL') or "postgresql://localhost/flaskvtube"
db_uri = "sqlite:///" + os.path.join(app.root_path, 'flaskvtube.db') # 追加
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
db = SQLAlchemy(app) 

class Entry(db.Model): 
    __tablename__ = "rooms4" 
    room_name = db.Column(db.String(), primary_key=True) 
    model_path = db.Column(db.String(), nullable=False) 
    background_path = db.Column(db.String(), nullable=False) 
    sound_path = db.Column(db.String(), nullable=False) 
    vmd_path = db.Column(db.String(), nullable=False) 
    subtitle_path = db.Column(db.String(), nullable=False)
    voice_path = db.Column(db.String(), nullable=False) 


def add_entry(room_name, model_path, background_path, sound_path, vmd_path, subtitle_path, voice_path):
    entry = Entry()
    entry.room_name = room_name
    entry.model_path = model_path
    entry.background_path = background_path
    entry.sound_path = sound_path
    entry.vmd_path = vmd_path
    entry.subtitle_path = subtitle_path
    entry.voice_path = voice_path

    db.session.add(entry)
    db.session.commit()
    return 0

def update_entry(room_name, model_path, background_path, sound_path, vmd_path, subtitle_path, voice_path):
    entry = Entry().query.filter(Entry.room_name == room_name).first()
    if model_path != None:
        entry.model_path = model_path
    if background_path != None:
        entry.background_path = background_path
    if sound_path != None:
        entry.sound_path = sound_path
    if vmd_path != None:
        entry.vmd_path = vmd_path
    if subtitle_path != None:
        entry.subtitle_path = subtitle_path
    if voice_path != None:
        entry.voice_path = voice_path   # entryがNone

    db.session.add(entry)
    db.session.commit()
    return 0


# 実行
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

# ※entry=一連の処理
