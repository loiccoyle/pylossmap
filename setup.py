# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pylossmap', 'pylossmap.header_handler']

package_data = \
{'': ['*'],
 'pylossmap': ['metadata/*',
               'metadata/custom_headers/*',
               'metadata/dcum/*',
               'metadata/headers/*']}

install_requires = \
['matplotlib>=3.2.1,<4.0.0',
 'pandas>=1.0.3,<2.0.0',
 'pytimber>=3.0.0,<4.0.0',
 'scipy>=1.4.1,<2.0.0',
 'tqdm>=4.44.1,<5.0.0']

extras_require = \
{'docs': ['sphinx<3.3.0',
          'sphinx-autoapi>=1.5.0,<2.0.0',
          'sphinx-rtd-theme>=0.5.0,<0.6.0',
          'm2r2>=0.2.5,<0.3.0'],
 'notebooks': ['jupyter>=1.0.0,<2.0.0']}

entry_points = \
{'console_scripts': ['header_maker = pylossmap.header_handler.cli:main']}

setup_kwargs = {
    'name': 'pylossmap',
    'version': '0.1.0',
    'description': '',
    'long_description': "[![Documentation Status](https://readthedocs.org/projects/pylossmap/badge/?version=latest)](https://pylossmap.readthedocs.io/en/latest/?badge=latest)\n\n# pylossmap\n\nThis library facilitates the fetching and handling of the LHC's BLM measurements.\n\n# Installation\n\n```sh\ngit clone https://github.com/loiccoyle/pylossmap\ncd pylossmap\npip install .\n```\n\n# Requirements\n\n* [pytimber](https://www.github.com/rdemaria/pytimber)\n* pandas\n* numpy\n* matplotlib\n\nOptional:\n* tqdm\n\n# Usage\n\n**See examples in the** `notebooks` **folder.**\n\nThere are 3 main classes:\n\n## The BLMDataFetcher class\n\nThis class will handle the fetching of BLM data and assigning the correct header information.\n\nIt has 2 main methods of fetching data:\n\n* `from_datetime`: which takes 2 datetime objects or epoch/unix time `int`s/`float`s.\n* `from_fill`: which takes a fill number along with the requeted beam modes.\n\nIt also has a helper method to fetch data surrounding triggers of the ADT blowup:\n\n* `iter_from_adt`: iteratively yield BLMData instances of data surrounding the trigger of the ADT blowup within the requested time range.\n\nA helper method to facilitate the fetching of BLM background for data following an ADT trigger:\n\n* `bg_from_ADT_trigger`: takes datetime of the ADT trigger, it will figure out a time interval where no ADT blowup triggers occurred and fetches the background data.\n\nThe fetcher class returns the data in the shape of a BLMData instance.\n\n## The BLMData class\n\nThis class handles the BLM data, the main methods are:\n\n* `plot`: creates a waterfall plot of the BLM data.\n* `iter_max`: iterates on index of the max values of the desired BLMS, defaults to the primary IR 7 BLMs.\n* `fetch_intensity`,`fetch_filling_scheme`,`fetch_number_bunches`,`fetch_energy`: fetches some additional beam information for the current time range.\n\nYou can access the raw data through the `data` attribute.\n\nIn order to create a loss map of a given time, use the `loss_map` method, and provide the desired datetime. This outputs a `LossMap` instance.\n\n## The LossMap class\n\nThis class handles the loss map processing. It provides a pandas/numpy style interface for filtering and selecting BLMs.\nSome main methods are:\n\n* `plot`: to create a loss map plot.\n* `set_background`: to set another LossMap instance as the background signal.\n* `clean_background`: to substract the background.\n* `normalize`: to normalize the data to the max value, or the signal of the provided BLM.\n* `fetch_intensity`,`fetch_filling_scheme`,`fetch_number_bunches`,`fetch_energy`: fetches some additional beam information for the current timestamp.\n* various methods for filtering based on `cell`, `IR`, `side`, collimator `type`, `beam`, ...\n",
    'author': 'Loic Coyle',
    'author_email': 'loic.coyle@hotmail.fr',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/loiccoyle/pylossmap',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.6.2,<4.0.0',
}


setup(**setup_kwargs)

