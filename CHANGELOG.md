# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-25

### Added

- Core evaluation pipeline with runner, prompt builder, and task loader
- AEI (Adaptation Efficiency Index) calculation for measuring few-shot learning efficiency
- Few-shot collapse detection supporting three types: few-shot collapse (formerly "negative learning"), peak regression, and mid-curve dip
- Basic scoring system with text scorers and LLM judge
- Model client support for Anthropic, Google GenAI, Vertex AI, LM Studio, and OpenAI-compatible APIs
- Streamlit viewer for learning curves and collapse detection visualization
- Demo task pack with 4 built-in tasks
- CLI output with metrics summary

[Unreleased]: https://github.com/ShuntaroOkuma/adapt-gauge-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ShuntaroOkuma/adapt-gauge-core/releases/tag/v0.1.0
