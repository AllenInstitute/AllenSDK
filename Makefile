PROJECTNAME = allen_wrench
DISTDIR = dist

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME)
	cp -r allen_wrench setup.py allen_wrench.egg-info README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz $(PROJECTNAME)

clean:
	rm -rf $(DISTDIR)
