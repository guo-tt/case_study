REPO := tiangg/case-study
TAG := 0.0.0

.PHONY: test deploy

clean:
	find . \( -name \*.pyc -o -name \*.pyo -o -name __pycache__ \) -prune -exec rm -rf {} +

build: clean
	docker build -f Dockerfile -t $(REPO):$(TAG) .

push:
	docker push $(REPO):$(TAG)

test:
	docker run -it -v /Users/guotiantian/Repo/case_study/data:/apps/data \
	--rm --name case_study \
	-p 8888:8888 \
	-v `pwd`:/notebooks \
	$(REPO):$(TAG) \
	tail -f /etc/hosts

# jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/notebooks --NotebookApp.token='' --NotebookApp.password=''

remove:
	sudo docker rmi $(REPO):$(TAG)