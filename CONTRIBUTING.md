# Contributing to the AllenSDK

Thank you for your interest in contributing!

### Bug reports and feature requests

* Before reporting a bug or requesting a feature, use Github's issue search to see if anyone else has already done so.
* If there is no existing issue, create a new one. You should include:
    * A brief, descriptive title
    * A clear description of the problem or request.
    * If you are reporting a bug, your description should contain the following information:
        * What you did (preferably the actual code or commands that you ran)
        * What happened
        * What you were expecting to happen
        * How your system is configured (operating system, Python version)
* If you are comfortable addressing this issue yourself, take a look at the [guide to contributing code](#code) below.

### Questions 

* Is your question about the Allen _Software Development Kit_, or about Allen Institute data and tools more generally?
    * If the latter, you should check our [online help](http://help.brain-map.org). If you don't find what you are looking for there you can submit your question using [this form](http://allins.convio.net/site/PageServer?pagename=send_us_a_message_ai).
* If you do have an AllenSDK question, first check the [documentation](http://alleninstitute.github.io/AllenSDK/index.html) (including the [examples](http://alleninstitute.github.io/AllenSDK/examples.html) and the [api reference](http://alleninstitute.github.io/AllenSDK/allensdk.html)) to make sure that your question is not already addressed.
    * If you can't find an answer in the documentation, create an issue on Github.

### Code

Code contributions should be submitted in the form of a pull request. Here are the steps:

* Make sure that there is an issue tracking your work. See [above](#bug-reports-and-feature-requests) for guidelines on creating effective issues.
* Create a [fork](https://help.github.com/articles/fork-a-repo/) of the AllenSDK and clone it locally by running
```
git clone <the url of your fork>
```
Which branch should you base your work on? See [below](#branches) for a detailed answer (the non-detailed answer is "probably `master`")
* Start working! Make sure to tag the issue in your commits by including `#issue_number` in the commit messages. This will help make the larger purpose of your changes clear. 
Also, every time you push to your fork, the issue page will be updated with a link to the commit, so that others can see the progress of your work.
* Run the tests! You can do this by running `make test` or `python -m pytest <a test file or directory>`. If you use the command in the Makefile, an html coverage report will be generated in the directory `htmlcov`.
**If you are adding functionality, you must also add tests.**
* When you are ready, create a pull request from your fork to the main repository. The title should be brief and descriptive, 
and the body of the pull request should include the phrase `Resolves #issue_number`, as well as a short description of the changes that you have implemented.
    * Before creating a pull request, make sure that your changes do not conflict with work in the main repository by [rebasing or merging](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) from the main repository into your fork.

###### Branches

There are two branches that you might want to base your work on:`master` and `internal`. 
The distinction comes down to whether your changes depend on API features or data that have not yet been publicly released by the Allen Institute for Brain Science.
If the answer to this question is "yes," you should branch from `internal`. Otherwise, branch from `master`.

Whenever a feature branch is merged into `master`, `master` should be merged into `internal` to ensure that the branches do not diverge. 
Merges in the other direction, from `internal` to `master`, will be performed during Allen Institute data releases, when the new API features and data become publically available.
