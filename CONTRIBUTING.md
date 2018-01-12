# Contributing to the AllenSDK


### Bug reports and feature requests

* Before reporting a bug or requesting a feature, use Github's issue search to see if anyone else has already done so.
* If there is no existing issue, create a new one. You should include:
        * A brief, descriptive title
        * A clear description of the problem or request. 

###### Writing bug reports

Please include the following information when reporting a bug:

* What you did (preferably the actual code or commands that you ran)
* What happened
* What you were expecting to happen
* How your system is configured (operating system, Python version)

### Questions 




### Code contributions

Code contributions should be submitted in the form of a pull request. To 

###### Branches

There are two branches that you might want to base your work on: *master* and *internal*. 
The distinction comes down to whether your changes depend on API features or data that have not yet been publically released by the Allen Institute for Brain Science.
If the answer to this question is "yes," you should branch from internal. Otherwise, branch from master.

Whenever a feature branch is merged into master, master should be merged into internal. 
Merges in the other direction, from internal to master, should be performed when 
