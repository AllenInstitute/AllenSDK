PROJECTNAME = allensdk
DISTDIR = dist
BUILDDIR = build
export RELEASE=dev$(BUILD_NUMBER)
RELEASEDIR = $(PROJECTNAME)-$(VERSION).$(RELEASE)
EGGINFODIR = $(PROJECTNAME).egg-info
DOCDIR = doc

DOC_URL=http://alleninstitute.github.io/AllenSDK
#ZIP_FILENAME=AllenSDK-master.zip
#TGZ_FILENAME=AllenSDK-master.tar.gz
#ZIP_URL=https:\/\/github.com\/AllenInstitute\/AllenSDK\/archive\/master.zip
#TGZ_URL=https:\/\/github.com\/AllenInstitute\/AllenSDK\/archive\/master.tar.gz

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME)
	cp -r allensdk setup.py README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz $(PROJECTNAME)
	

distutils_build: clean
	python setup.py build
	
setversion:
	sed -i --expression 's/'\''[0-9]\+.[0-9]\+.[0-9]\+'\''/'\''${VERSION}.${RELEASE}'\''/g' allensdk/__init__.py

sdist: distutils_build
	python setup.py sdist

EXAMPLES=doc/_static/examples

doc: FORCE
	sphinx-apidoc -d 4 -H "Allen SDK" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION).dev$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc
	cp -R doc_template/examples $(EXAMPLES)
	sed -i --expression "s/|version|/${VERSION}/g" doc/conf.py
	cp -R doc_template/aibs_sphinx/static/* doc/_static
	cp -R doc_template/aibs_sphinx/templates/* doc/_templates
	cd doc && find . -name '*.rst' -exec sed -i --expression "s/|tgz_url|/${TGZ_URL}/g" {} \;
	cd doc && find . -name '*.rst' -exec sed -i --expression "s/|zip_url|/${ZIP_URL}/g" {} \;	
	cd doc && find . -name '*.rst' -exec sed -i --expression "s/|tgz_filename|/${TGZ_FILENAME}/g" {} \;
	cd doc && find . -name '*.rst' -exec sed -i --expression "s/|zip_filename|/${ZIP_FILENAME}/g" {} \;
	cd $(EXAMPLES)/docker && find . -name 'Dockerfile.*' -exec sed -i --expression "s/|tgz_filename|/${TGZ_FILENAME}/g" {} \;
	cd $(EXAMPLES)/docker && find . -name 'Dockerfile.*' -exec sed -i --expression "s/|tgz_url|/${TGZ_URL}/g" {} \;
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/layout.html
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/portalHeader.html
	sed -i --expression "s/\/external_assets/${STATIC}\/external_assets/g" doc/_static/external_assets/javascript/portal.js
	cd $(EXAMPLES)/nb && find . -maxdepth 1 -name '*.ipynb' -exec jupyter nbconvert --to html {} \;
	cd $(EXAMPLES)/nb/friday_harbor && find . -maxdepth 1 -name '*.ipynb' -exec jupyter nbconvert --to html {} \;
	cd doc && make html || true

FORCE:

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
