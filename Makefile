setup:
	py -m venv venv

install_old:
	./venv/Scripts/activate && py -m pip install -r requirements.txt

install:
	py -m pip install -r requirements.txt

run:
	cd nsduh_rag && py nsduh_query.py
