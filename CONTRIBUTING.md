# Contributing to the AllenSDK

Thank you for your interest in contributing! There are a few ways you can contribute
to the AllenSDK project:

* [Answering User Questions](#answering-user-questions)
* [Adding test coverage](#testing-guidelines)
* [Add Example Notebooks](#add-example-notebooks)
* [Reporting Bugs](#reporting-bugs)
* [Requesting features](#suggesting-featuresenhancements)
* [Contributing Code](#contributing-code)

## Answering User Questions
A great way to contribute to the AllenSDK is to help answer user questions on 
the [forums](https://community.brain-map.org) or [StackOverflow](https://stackoverflow.com/questions/tagged/allen-sdk). Assisting new users is not only a valuable service for the
community, but also contributes to a wider culture of scientific collaboration.

## Bug Reports and Feature Requests

Before reporting a bug or requesting a feature, use Github's issue search to see if anyone else has already done so. Don't reopen an issue tagged with `wontfix` without getting consensus
from the AllenSDK team in the comment thread.

### Reporting Bugs 
If there is no existing issue, create a new one. You should include:
* A brief, descriptive title
* A clear description of the problem 
* If you are reporting a bug, your description should contain the following information:
    * What you did (preferably the actual code or commands that you ran)
    * What happened
    * What you were expecting to happen
    * How your system is configured (operating system, Python version)

If you are comfortable addressing this issue yourself, take a look at the [guide to contributing code](#contributing-code) below.

### Add Example Notebooks
Adding example notebooks are a great way to contribute to documentation and help
other users get started on the AllenSDK. Take a look at our [existing notebooks](https://github.com/AllenInstitute/AllenSDK/tree/master/doc_template/examples_root/examples/nb)
as a general guide for the style and content. We have many great notebooks, but 
[Extracellular Electrophysiology Data](https://github.com/AllenInstitute/AllenSDK/blob/master/doc_template/examples_root/examples/nb/ecephys_session.ipynb) is a good one to take a look at to get a sense for what we're looking for. All new notebook contributions should be compatible with Python 3.6+.

Notebook Guidelines:

* Provide notebooks for major features/interfaces.
* Use markdown cells for titles, explanatory text, and subtitles that clarify your code. 
* Include external references additional resources as needed.
* Import your libraries in the first code cell.
* Display your graphics inline -- for example, using the magic command `%matplotlib inline`.
* Try to keep the cells of your notebook fairly simple. Think of a cell as a paragraph in a book; its contents should support a single idea.
* Make sure all of the cells are required and *in order* -- you should be able
to execute all cells in the notebook from the top down.
* *Check in your cell outputs.* We do not currently build notebooks when we generate documentation due to the long runtime for some examples.
* Save your notebook in `doc_template/examples_root/examples/nb`.

### Suggesting Features/Enhancements
Before suggesting a feature or enhancement, please check existing issues as you may find out
you don't need to create one. When you create an enhancement suggestion, please include
as many details as possible in the issue template.

When contributing a new feature to the AllenSDK, the maintenance burden is (by default)
transferred to the AllenSDK team. This means that the benefit of the contribution must be
weighed against the cost of maintaining the feature. 

When suggesting a feature, consider:
* Is the change clearly explained and motivated?
* Would the enhancement be useful for most users?
* Is this a new feature that can stand alone as a third party project?
* How does this change impact existing users?

### Asking Questions 

* Is your question about the Allen _Software Development Kit_, or about Allen Institute data and tools more generally?
    * If the latter, you should check our [online help](http://help.brain-map.org) or the [Allen Brain Map Community Forum](https://community.brain-map.org). If you can't find what you are looking for with the aforementioned resources, you can submit your question using [this form](http://allins.convio.net/site/PageServer?pagename=send_us_a_message_ai).
* If you do have an AllenSDK question, first check the [documentation](http://alleninstitute.github.io/AllenSDK/index.html) (including the [examples](http://alleninstitute.github.io/AllenSDK/examples.html) and the [api reference](http://alleninstitute.github.io/AllenSDK/allensdk.html)) to make sure that your question is not already addressed.
    * If you can't find an answer in the documentation, please create an issue on Github.

## Contributing Code

If you are able to improve the AllenSDK, send us your pull requests!
Contributing code yourself can be a great way to include the features you want
in the AllenSDK.

To contribute code, please follow this list:
* Read [contributing guidelines](#contributing-to-the-allensdk)
* Sign Contributor License Agreement (CLA)
* Check if changes are consistent with [Coding Style](#style-guidelines)
* [Write Unit Tests](#testing-guidelines)
* Run Unit Tests 
* Make a pull request

### Deciding What to Contribute

Navigate to the Github ["issues"](https://github.com/AllenInstitute/AllenSDK/issues?q=is%3Aopen+is%3Aissue) 
tab and start looking through
issues. The AllenSDK team uses Github issues to track our internal development, so we 
recommend filtering to issues with the "good first issue" label or issues with the
"help wanted" label. These are issues that we believe are particularly well
suited for outside contributions, often because we won't get to them right away.
If you decide to start on an issue, leave a comment so that other people know that 
you are working on it.

### Setting Up

Code contributions should be submitted in the form of a pull request. Here are the steps:

* Make sure that there is an issue tracking your work. See [above](#bug-reports-and-feature-requests) for guidelines on creating effective issues.
* Create a [fork](https://help.github.com/articles/fork-a-repo/) of the AllenSDK and clone it to your development environment.

* Make a new branch for your code off of `master`. For consistency and use with 
visual git plugins, we prefer the following convention for branch naming:
`GH-<issue-number>/<bugfix/feature>/<short-description>`. For example:
    ```
    GH-712/bugfix/auto-reward-key
    GH-9999/feature/parallel-behavior-analysis
    ```
* Create an environment and install necessary requirements: `requirements.txt` and `test_requirements.txt`
* Start writing code!

### Style Guidelines
We follow [PEP-8 guidelines](https://www.python.org/dev/peps/pep-0008/) for new python code.
We also follow [PEP-484](https://www.python.org/dev/peps/pep-0484/) for type annotations.
Before submitting a pull request, run [flake8](https://pypi.python.org/pypi/flake8/) and 
[mypy](https://pypi.org/project/mypy/) linters to check the style of your code. All new code contributions should be compatible with Python 3.6+.

#### Docstrings
Docstrings for new code should follow the [Numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html). This allows us to ensure consistency in our auto-generated API documentation.

### Testing Guidelines
All code you write should have unit tests, including bugfixes (since the presence of bugs
likely indicates a gap in test coverage). We use [pytest](https://docs.pytest.org/en/latest/) 
for running unit tests.

If you write a new file `foo.py`, you should place its unit tests in `test_foo.py`.
Follow the directory structure of the parent module(s) for your tests so that 
they are easy to find. For example, tests for `allensdk/brain_observatory/foo.py`
should be in `allensdk/test/brain_observatory/test_foo.py`.

**Testing Guidelines**
* Smaller, faster tests are better (and more likely to be run!)
* Tests should be deterministic
* Tests should be hermetic. They should be packed with everything they need and start any fake services they might need.
* Tests should work every time; use dependency injection to mock out flaky or long-running services.

### Committing Guidelines
Commit messages should have a subject line, separated by a blank line and then 
paragraphs of approximately 72 char lines. For the subject line, shorter is better --
ideally 72 characters maximum. In the body of the commit message, more detail
is better than less. See [Chris Beams](https://chris.beams.io/posts/git-commit/) for
more guidelines about writing good commit messages.

* Tag the issue number in your subject line. For Github issues, it's helpful to 
use the abbreviation ("GH") to separate it from Jira tickets.
    ```
    GH #1111 - Add commit message guidelines

    This contains more detailed information about the feature
    or bugfix. It's written in complete sentences. It has
    appropriate capitalization and punctuation. It's separated
    from the subject by a blank line.
    ```
* Limit commits to the most granular changes that make sense. Group together small
units of work into a single commit when applicable. Think about readability;
your commits should tell a story about your changes that everyone can follow. 

### Making a Pull Request
* Make sure your tests pass locally first (`make test` or `python -m pytest <a test file or directory>`)
* Update your forked repository and rebase your branch onto the latest `master` branch.
* Target the latest release candidate branch for your PR. This branch has the format `rc/x.y.z`.
* Use a brief but descriptive title.
* Include `Relates to: #issue_number` and a short description of your changes in the
body of the pull request.
* Support your changes with additional resources. Having an example notebook
or visualizations can be very helpful during the review process.

### Review Process
Once your pull request has been made and your tests are passing, a member of the AllenSDK
team will be assigned to review. Please be patient, as it can take some time to 
assign team members. Once your pull request has been approved, the AllenSDK
team member will merge your changes into the latest release candidate branch.
Your changes will be included in the next release cycle. Releases typically
occur every 2-4 weeks.

If in doubt how to do anything, don't hesitate to ask a team member!