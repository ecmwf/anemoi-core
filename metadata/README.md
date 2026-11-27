# anemoi-metadata

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/incubating_badge.svg" alt="Maturity Level">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <!-- documentation badges -->
  <a href="https://anemoi.readthedocs.io/projects/metadata/en/latest/">
    <img src="https://img.shields.io/readthedocs/anemoi-metadata/latest?label=Docs%20(Metadata)&color=green" alt="Anemoi metadata Docs">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Incubating** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).


**DISCLAIMER**
This project is **BETA** and will be **Experimental** for the foreseeable future.
Interfaces and functionality are likely to change, and the project itself may be scrapped.
**DO NOT** use this software in any project/software that is operational.

Read and write the metadata stored inside Anemoi checkpoints.

Checkpoints have been written in several metadata formats. `anemoi-metadata` opens all of them through
one interface (`Metadata`), reads leniently, and checks the metadata strictly when it writes it.

```python
from anemoi.metadata import Metadata

metadata = Metadata.from_checkpoint("model.ckpt")
print(metadata.version, metadata.component_versions)
print(metadata.task.timestep())
dataset = metadata.dataset("data")  # the name may be omitted for a single-dataset checkpoint
print(dataset.data.grid_size(), dataset.arrays.latitudes)
```

A command line tool is installed too: `anemoi-metadata info model.ckpt`.

## Documentation

The documentation can be found at https://anemoi.readthedocs.io/projects/metadata/en/latest/.

## Install

Install via `pip` with:

```
$ pip install anemoi-metadata
```

## License

```
Copyright 2026-, Anemoi contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
