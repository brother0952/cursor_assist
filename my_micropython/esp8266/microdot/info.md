# microdot 源码自带demo，有控制gpio的


'''python
@app.route('/')
def index(request):
    return 'Hello, world!'


@app.route('/users/active')
def active_users(request):
    return 'Active users: Susan, Joe, and Bob'


@app.route('/invoices', methods=['GET', 'POST'])
def invoices(request):
    if request.method == 'GET':
        return 'get invoices'
    elif request.method == 'POST':
        return 'create an invoice'
        

@app.route('/invoices', methods=['GET'])
def get_invoices(request):
    return 'get invoices'

@app.route('/invoices', methods=['POST'])
def create_invoice(request):
    return 'create an invoice'
    
    
@app.get('/invoices')
def get_invoices(request):
    return 'get invoices'

@app.post('/invoices')
def create_invoice(request):
    return 'create an invoice'
    
    
@app.get('/users/<username>')
def get_user(request, username):
    return 'User: ' + username
    
    
@app.get('/users/<firstname>/<lastname>')
def get_user(request, firstname, lastname):
    return 'User: ' + firstname + ' ' + lastname


@app.get('/users/<int:id>/<string:username>')
def get_user(request, id, username):
    return 'User: ' + username + ' (' + str(id) + ')'


@app.get('/tests/<path:path>')
def get_test(request, path):
    return 'Test: ' + path



@app.get('/users/<re:[a-zA-Z][a-zA-Z0-9]*:username>')
def get_user(request, username):
    return 'User: ' + username


@app.before_request
def authenticate(request):
    user = authorize(request)
    if not user:
        return 'Unauthorized', 401
    request.g.user = user


@app.before_request
def start_timer(request):
    request.g.start_time = time.time()

@app.after_request
def end_timer(request, response):
    duration = time.time() - request.g.start_time
    print(f'Request took {duration:0.2f} seconds')


@app.errorhandler(404)
def not_found(request):
    return {'error': 'resource not found'}, 404


@app.errorhandler(ZeroDivisionError)
def division_by_zero(request, exception):
    return {'error': 'division by zero'}, 500


@app.get('/shutdown')
def shutdown(request):
    request.app.shutdown()
    return 'The server is shutting down...'



from microdot import redirect

@app.get('/')
def index(request):
    return redirect('/about')


     
     
     
from microdot import send_file

@app.get('/')
def index(request):
    return send_file('/static/index.html')


from microdot import send_file

@app.get('/')
def image(request):
    return send_file('/static/image.jpg', max_age=3600)  # in seconds


@app.route('/static/<path:path>')
def static(request, path):
    if '..' in path:
        # directory traversal is not allowed
        return 'Not found', 404
    return send_file('static/' + path, max_age=86400)






      

'''