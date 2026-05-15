
# Hints

## Integration of GPI using Git

The GPI is included as a Git *subtree* (NOT *submodule*).
This feature allows to nest a repository inside another one as a subdirectory,
while still managing project dependencies in a clean way.
In contrast to submodules, subtrees do not require special handling from users
of the super-project that do not actively touch the sub-projects. Indeed, they
can ignore the fact that we are using git subtree to manage dependencies
(code is contained in and available right after cloning the super-project).
For details, see e.g.
* https://manpages.debian.org/testing/git-man/git-subtree.1.en.html (manpage)
* https://www.youtube.com/watch?v=sC1sfoCo5qY (Video)
* https://www.atlassian.com/git/tutorials/git-subtree

Take care with the following links, which (in Mar 2025) discussed an older way of 
implementing similar functionality that you should not use anymore.
* https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging (corresponding section in "Pro Git" book 2nd edition)
* https://docs.github.com/en/get-started/using-git/about-git-subtree-merges

### Setup

This has already been done (you do NOT need to do that again), just for documentation.

1. Add the sub-project as a remote (not strictly necessary, but simplifies usage):\
   ``git remote add -f gpi git@github.com:nes-lab/gpi.git``

2. Add git subtree at a specified prefix folder:\
   ``git subtree add --prefix src/gpi gpi main --squash``

   `gpi` ist the name of the remote (defined in 1.), `main` is the name of the upstream branch.
   The ``--squash`` option is needed to avoid storing the entire history of the sub-project
   in the super-project (this is common practice to avoid messing up the super-project's history).
   The command generates a squash commit (squashing the history of the sub-project)
   and a merge commit (merging the new subdirectory into the super-project).

### How to update GPI from its upstream repository

To merge updates from the sub-project's upstream repository use the following commands.

1. ``git fetch gpi main``

2. ``git subtree pull --prefix src/gpi gpi main --squash``\
   Alternatively, you can use ``git subtree merge --prefix src/gpi --squash <from-commit>`` to merge manually.

Note: When using ``--squash``, the merge direction does not necessarily have to be forward,
you can also go back to a previous commit.

### Contributing changes upstream

One can freely commit fixes in the subdirectory to the super-project. 

To contribute changes to the upstream repository of the sub-project, use\
``git subtree push --prefix src/gpi gpi main``

To get clean commit messages, do not mix changes in super- and sub-projects in the same commit.

### Other commands

``git subtree split --prefix <prefix>`` can be used to extract a new sub-project from an existing
subdirectory together with a synthetic history extracted from the super-project (collecting all 
entries from the super-project that affected the <prefix> subtree).

