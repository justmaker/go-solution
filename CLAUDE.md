# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based solution repository for algorithm and programming problems.

## Common Commands

```bash
# Initialize module (if go.mod doesn't exist yet)
go mod init go-solution

# Run all tests
go test ./...

# Run tests in a specific package
go test ./path/to/package

# Run a single test by name
go test ./path/to/package -run TestName

# Run tests with verbose output
go test -v ./...

# Run benchmarks
go test -bench=. ./...

# Build
go build ./...

# Format code
gofmt -w .

# Vet (static analysis)
go vet ./...
```

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Use Go modules for dependency management
