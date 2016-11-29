PROJECTNAME = allensdk
DISTDIR = dist
BUILDDIR = build
export RELEASE=dev$(BUILD_NUMBER)
RELEASEDIR = $(PROJECTNAME)-$(VERSION).$(RELEASE)
EGGINFODIR = $(PROJECTNAME).egg-info
DOCDIR = doc
COVDIR = htmlcov

DOC_URL=http://alleninstitute.github.io/AllenSDK
#ZIP_FILENAME=AllenSDK-master.zip
#TGZ_FILENAME=AllenSDK-master.tar.gz
#ZIP_URL=https:\/\/github.com\/AllenInstitute\/AllenSDK\/archive\/master.zip
#TGZ_URL=https:\/\/github.com\/AllenInstitute\/AllenSDK\/archive\/master.tar.gz

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME) 
	cp -r allensdk setup.py README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz --exclude .git $(PROJECTNAME)
	

distutils_build: clean
	python setup.py build
	
setversion:
	sed -i --expression 's/'\''[0-9]\+.[0-9]\+.[0-9]\+'\''/'\''${VERSION}.${RELEASE}'\''/g' allensdk/__init__.py

sdist: distutils_build
	python setup.py sdist

pypi_register:
	python setup.py register --repository https://testpypi.python.org/pypi

pypi_deploy:
	python setup.py sdist upload --repository https://testpypi.python.org/pypi

pytest:
	find -L . -name "test_*.py" -exec py.test --boxed --pep8 --cov-config coveragerc --cov=allensdk --cov-report html --assert=reinterp --junitxml=test-reports/test.xml {} \+

pytest_lax:
	find -L . -name "test_*.py" -exec py.test --boxed --cov-config coveragerc --cov=allensdk --cov-report html --assert=reinterp --junitxml=test-reports/test.xml {} \+

pytest_lite:
	find -L . -name "test_*.py" -exec py.test --boxed --assert=reinterp --junitxml=test-reports/test.xml {} \+

pylint:
	pylint --disable=C allensdk > htmlcov/pylint.txt || exit 0
	grep import-error htmlcov/pylint.txt > htmlcov/pylint_imports.txt

flake8:
	flake8 --ignore=E201,E202,E226 --max-line-length=200 allensdk | grep -v "local variable '_' is assigned to but never used" > htmlcov/flake8.txt
	grep "imported but unused" htmlcov/flake8.txt > htmlcov/imports.txt

EXAMPLES=doc/_static/examples

doc: FORCE
	sphinx-apidoc -d 4 -H "Allen SDK" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION).$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc
	cp -R doc_template/examples $(EXAMPLES)
	sed -i --expression "s/|version|/${VERSION}/g" doc/conf.py
	cp -R doc_template/aibs_sphinx/static/* doc/_static
	cp -R doc_template/aibs_sphinx/templates/* doc/_templates
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/layout.html
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/portalHeader.html
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_static/external_assets/javascript/portal.js
	cd $(EXAMPLES)/nb && find . -maxdepth 1 -name '*.ipynb' -exec jupyter-nbconvert --to html {} \;
	cd $(EXAMPLES)/nb/summer_workshop_2015 && find . -maxdepth 1 -name '*.ipynb' -exec jupyter-nbconvert --to html {} \;
	cd doc && make html || true

FORCE:

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
	rm -rf $(COVDIR)
