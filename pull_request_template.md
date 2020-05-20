<!--Thank you for contributing to AllenSDK, your work and time will help to
advance open science!-->

# Overview:
<!-- Give a brief overview of the issue you are solving. Succinctly
explain the GitHub issue you are addressing and the underlying problem
of the ticket
example: 
Science team is not able to load max or avg projections for ophys
session #blah. A ITK image cannot be created because input resolution is (0,0).
It was found through investigation that LIMs was returning a 0 pixel resolution
for this behavior session.-->

# Addresses:
<!-- Add a link to the issue on Github board
example: 
Addresses issue [#1234](git_hub_ticket_url)-->

# Type of Fix:
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing
      functionality to not work as expected)
- [ ] Documentation Change

# Solution:
<!-- Outline your solution to the previously described issue and
underlying cause. This section should include a brief description of
your proposed solution and how it addresses the cause of the ticket
example:
Solution to this problem is to update the value of the pixel resolution to
0.78125 (scientifica resolution) if pixel resolution is returned as 0. This 
will address the underlying problem by providing a fallback value if the data
is not available from LIMs to query. A downfall is if default resolution is disparate
from actual resolution that wasn't saved, images might appear very distorted.
An alternative solution is to update the LIMs db to have the correct data for 
all microscope sessions.-->

# Adds:
<!-- Include a bulleted list or check box list of the implemented changes
in brief, as well as the addition of supplementary materials(unit tests,
integration tests, etc
example:
- Check for 0 pixel resolution coming from LIMs
- Assignment of default scientifica value of 0.78125 in case of zero return
- Unit tests for the resolution gettr function to test for various edge cases
-->

# Validation:
<!-- Describe how you have validated that your solution addresses the
root cause of the ticket. What have you done to ensure that your
addition is bug free and works as expected. Please provide specific
instructions so we can reproduce and list any relevant details about
your configuration
example:
- Screenshot of max projection from failing session
- Screenshot of avg projection from failing session
- Screenshot of passing unit tests
- Description of unit test cases
- Attached script to create max and avg projections of behavior session
- Windows 10.x.x.x, Surface Book 2 baseline, Conda Version 1.x.x-->
### Screenshots:
### Unit Tests:
### Script to reproduce error and fix:
### Configuration details:

# Checklist
- [ ] My code follows [Allen Institute Contribution Guidelines](https://github.com/AllenInstitute/AllenSDK/blob/master/CONTRIBUTING.md)
- [ ] My code is unit tested where appropriate and does not decrease
      test coverage
- [ ] I have performed a self review of my own code
- [ ] My code is well-documented, and the docstrings conform to [Numpy Standards](https://numpydoc.readthedocs.io/en/latest/format.html)
- [ ] I have updated the documentation of the repository where
      appropriate
- [ ] The header on my commit includes the issue number

# Notes:
<!-- Use this section to add anything you think worth mentioning to the
reader of the issue
example:
I noticed that values from LIMs query for pixel resolution are returning zero
I have made a new issue to address this error at #5678. I believe this is an 
error as all sessions should have a pixel resolution provided by the microscope.-->
