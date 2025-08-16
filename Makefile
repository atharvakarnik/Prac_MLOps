install:
	pip install --upgrade pip &&\
		pip install -r Ch1/requirements.txt
lint:
	pylint --disable=R,C Ch1/hello.py

test:
	python -m pytest -vv --cov=hello Ch1/test_hello.py