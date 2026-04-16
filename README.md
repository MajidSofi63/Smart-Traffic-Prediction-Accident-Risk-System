If You Get Error:
Check these in Render dashboard:

Environment Variables (Settings → Environment Variables):

Add: PYTHON_VERSION = 3.11.6

Add: FLASK_DEBUG = False

Start Command (Settings):

Should be: gunicorn app:app --timeout 120 --bind 0.0.0.0:$PORT

Build Command (Settings):

Should be: pip install -r requirements.txt
