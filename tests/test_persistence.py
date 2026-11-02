"""별도 Python 프로세스 재시작 뒤 DB와 결과 파일이 유지되는지 확인."""
import os
import subprocess
import sys
from pathlib import Path

def test_restart_persists_records_and_files(tmp_path):
    env=os.environ.copy()
    env['DATABASE_URL']='sqlite:///'+str(tmp_path/'persist.db')
    env['LOOK_STORAGE']=str(tmp_path/'files')
    root=Path(__file__).resolve().parents[1]
    create='''
from app.db import init_db, transaction, Input
from app.media import register_image
from app.jobs import analyze_image
from PIL import Image
import io
init_db()
b=io.BytesIO();Image.new('RGB',(320,240),'white').save(b,format='PNG')
iid=register_image(b.getvalue(),'restart.png')
analyze_image(iid)
with transaction() as s:
    assert s.query(Input).count()==1
'''
    read='''
from app.db import init_db, transaction, Input, Run, Inference
from app.media import path
init_db()
with transaction() as s:
    i=s.query(Input).one()
    r=s.query(Run).one()
    inf=s.query(Inference).one()
    assert r.status=='DONE'
    assert inf.payload['status']=='NO_HAND'
    assert path(i.original).is_file()
    assert path(inf.payload['image']).is_file()
'''
    for code in (create,read):
        r=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=30)
        assert r.returncode==0,r.stdout+r.stderr
