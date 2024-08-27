# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.11] - 2024-08-27
### Fixed
- Tag bug, PiPy upload was not working

## [0.0.10] - 2024-08-04
We fixed a bug with the Sydney model, where only the last batch would be returned during batchful evaluation, instead of the full data set.

## [0.0.9] - 2024-05-24
The API was changed to accept and yield sequences of arbitrary length.
It is assumed that a provided sequence describes a full period.
Internally, the different team models work with fix sequence lengths, so the user input is 1d-interpolated linearly accordingly.

## [0.0.8] - 2024-05-23
Bump to new subversion.

## [0.0.7] - 2024-05-23
### Fixed
Small bug fixes and allow pandas >= 2 instead of >= 2.0.3, in order to avoid requirements conflict in magnet-engine

## [0.0.6] - 2024-04-30

## [0.0.5] - 2024-04-13
### Added
 - Sydney model

## [0.0.4] - 2024-03-30
### Changed
 - Change python import package name to be magnethub to avoid underscores

## [0.0.3] - 2024-03-28
### Changed
 - Change pip installation name from mag_net_hub to mag-net-hub

## [0.0.2] - 2024-03-27
### Fixed
 - Missing requirements and missing models in pip package

## [0.0.1] - 2024-03-26
### Added
 - Paderborn Model 

[unreleased]: https://github.com/upb-lea/mag-net-hub/compare/0.0.11...HEAD
[0.0.11]: https://github.com/upb-lea/mag-net-hub/compare/v0.0.10...0.0.11
[0.0.10]: https://github.com/upb-lea/mag-net-hub/compare/0.0.9...v0.0.10
[0.0.9]: https://github.com/upb-lea/mag-net-hub/compare/0.0.8...0.0.9
[0.0.8]: https://github.com/upb-lea/mag-net-hub/compare/0.0.7...0.0.8
[0.0.7]: https://github.com/upb-lea/mag-net-hub/compare/0.0.6...0.0.7
[0.0.6]: https://github.com/upb-lea/mag-net-hub/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/upb-lea/mag-net-hub/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/upb-lea/mag-net-hub/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/upb-lea/mag-net-hub/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/upb-lea/mag-net-hub/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/upb-lea/mag-net-hub/releases/tag/0.0.1
