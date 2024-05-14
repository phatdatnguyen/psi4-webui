import os

os.environ.setdefault('PYTHONIOENCODING', 'utf8')

if os.environ.get('PYTHONIOENCODING', '').lower() not in {'utf-8', 'utf8'}:
    raise EnvironmentError("Environment variable $PYTHONIOENCODING must be set to 'utf8'")
else:
    print('UTF8')