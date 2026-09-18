# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from anemoi.utils.testing import TemporaryDirectoryForTestData


class GetTmpPath:
    def __init__(self, temporary_directory_for_test_data: TemporaryDirectoryForTestData) -> None:
        self.temporary_directory_for_test_data = temporary_directory_for_test_data

    def __call__(self, url: str) -> tuple[Path, str]:

        url_archive = url + ".tgz"
        name_dataset = Path(url).name
        tmp_path_dataset = self.temporary_directory_for_test_data(url_archive, archive=True)

        tmp_path = Path(tmp_path_dataset) / name_dataset

        return tmp_path, url_archive
