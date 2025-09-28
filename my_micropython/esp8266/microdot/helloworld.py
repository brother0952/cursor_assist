
from microdot import Microdot

app = Microdot()




@app.route('/')
def index(request):
    return 'jw Hello, world!'

app.run(host='0.0.0.0', port=80, debug=False, ssl=None)

