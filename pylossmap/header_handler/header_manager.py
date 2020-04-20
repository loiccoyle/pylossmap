import logging
import warnings
import pandas as pd

from os import remove
from shutil import copytree
from pathlib import Path

from ..utils import sanitize_t


class HeaderManager:
    def __init__(self,
                 header_folder=Path(__file__).parents[1] / 'metadata' / 'custom_headers'):
        self.header_folder = header_folder
        self._logger = logging.getLogger(__name__)

    # @staticmethod
    # def _sanitize_file_name(file_name):
    #     """Sanitize header file name.
    #     """
    #     return file_name.replace('-', ' ').replace(':', ' ').replace(' ', '')

    def select(self, t, **kwargs):
        """Select a header file based on the a given time.

        Args:
            t (int,float,str): see utils.sanitize_t.
            **kwargs: passed to pandas get_loc

        Raises:
            ValueError: if header folder is empty.

        """
        t = sanitize_t(t)
        # this warnings stuff is due to a bug in pandas:
        # https://stackoverflow.com/questions/54854900/workaround-for-pandas-futurewarning-when-sorting-a-datetimeindex
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            headers = self.list()
            if not headers:
                raise ValueError(f'Header folder {self.header_folder} is empty.')
            selected_i = pd.to_datetime([t.stem for t in headers],
                                        format='%Y_%m_%d_%H_%M_%S%z')\
                .get_loc(t, **kwargs)
        return headers[selected_i]

    def add(self, t, header, ignore_existing=False):
        """Adds a header file to the header folder.

        Args:
            t (int,float,str): see utils.sanitize_t.
            header (list): list of BLM names.
            ignore_existing (bool, optional): if True will override any already
                exisiting header at that timestamp.

        Raises:
            FileExistsError: If file already exists and ignore_existing is
                False.
        """
        t = sanitize_t(t)
        file = Path(self.header_folder / (t.strftime('%Y_%m_%d_%H_%M_%S%z') +
                                          '.csv'))

        # print(file)
        header = '\n'.join(header)
        if not file.exists() or ignore_existing:
            with open(file, 'w') as fp:
                fp.write(header)
                self._logger.info(f"Added {file} header file.")
        else:
            raise FileExistsError(f'File {file} already exists, pass '
                                  'ignore_existing=True to replace.')

    def remove(self, t, **kwargs):
        """Deletes a header file from the header folder.

        Args:
            t (int,float,str): see utils.sanitize_t.
            **kwargs: forwarded to pd.Index.get_loc().

        Raises:
            FileNotFoundError: If the file does not exists.
            ValueError: If there are no files in the header folder.
        """
        t = sanitize_t(t)
        selected_file = self.select(t, **kwargs)
        if selected_file.is_file():
            # print(f'removing {headers[selected_i]}')
            remove(selected_file)
            self._logger.info(f"Removed {selected_file} header file.")
        else:
            raise FileNotFoundError(f'File {selected_file} not found.')

    def read(self, t, **kwargs):
        t = sanitize_t(t)
        selected_file = self.select(t, **kwargs)
        if selected_file.is_file():
            with open(selected_file, 'r')as fp:
                return [l.rstrip() for l in fp]
        else:
            raise FileNotFoundError(f'File {selected_file} not found.')

    def list(self):
        """Lists the files in the header folder.

        Returns:
            list: list of files.
        """
        return list(sorted(self.header_folder.glob('*')))

    def export(self, destination=None):
        """Copies the header folder elsewhere.

        Args:
            destination (path like, optional): destination where to paste the
                folder, defaults to the current working directory.
        """
        if destination is None:
            destination = Path.cwd() / self.header_folder.name
        copytree(self.header_folder, destination)
        self._logger.info(f"Copied {self.header_folder} to {destination}.")
