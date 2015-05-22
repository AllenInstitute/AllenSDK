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
	sed -ie 's/'\''[0-9]\+.[0-9]\+.[0-9]\+'\''/'\''${VERSION}.${RELEASE}'\''/g' allensdk/__init__.py

sdist: distutils_build
	python setup.py sdist

doc: FORCE
	sphinx-apidoc -d 4 -H "Allen SDK" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION).dev$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc
	cp -R doc_template/examples doc
	sed -ie "s/|version|/${VERSION}/g" doc/conf.py
	cp -R doc_template/aibs_sphinx/static/* doc/_static
	cp -R doc_template/aibs_sphinx/templates/* doc/_templates
	sed -ie "s/|tgz_url|/${TGZ_URL}/g" doc/install.rst
	sed -ie "s/|tgz_url|/${TGZ_URL}/g" doc/links.rst
	sed -ie "s/|zip_url|/${ZIP_URL}/g" doc/links.rst
	sed -ie "s/|tgz_filename|/${TGZ_FILENAME}/g" doc/links.rst
	sed -ie "s/|zip_filename|/${ZIP_FILENAME}/g" doc/links.rst
	sed -ie "s/|tgz_url|/${TGZ_URL}/g" doc/examples/docker/Dockerfile.ubuntu
	sed -ie "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/layout.html	
	sed -ie "s/\/external_assets/${STATIC}\/external_assets/g" doc/_templates/portalHeader.html
	sed -ie "s/\/external_assets/${STATIC}\/external_assets/g" doc/_static/external_assets/javascript/portal.js
	cd doc && make html || true
	cp doc_template/.nojekyll doc/_build/html

FORCE:

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
