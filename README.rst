================
README for pyrwt
================

pyrwt is a python package that uses cython to wrap the
RICE Wavelet Toolbox. It is fast and enables comparing
the results of your python algorithm with the results
of algorithms that use the Matlab toolbox.

Installing
==========

Use ``setup.py``::

   python setup.py install


Reading the docs
================

After installing::

   cd doc
   make html

Then, direct your browser to ``build/html/index.html``.


Testing
=======

To run the tests with the interpreter available as ``python``, use::

   cd examples
   python denoise.py


Conditions for use
==================

See the LICENSE file


Contributing
============

For bug reports use the Bitbucket issue tracker.
You can also send wishes, comments, patches, etc. to amitibo@tx.technion.ac.il


Acknowledgement
===============

Thank-you to the people at <http://wingware.com/> for their policy of **free licenses for non-commercial open source developers**.

.. image:: http://wingware.com/images/wingware-logo-180x58.png
