# Who's that again?

## Introduction

A frustration for many readers is the difficulty of remembering who all the
characters of a book are and how they relate to one another. While some books
offer family trees or lists of characters, many books don't, even when the
dramatis personae is extensive. The goal of this project is to automate the
generation of a list of characters, establish which characters have close
relationships, and determine the nature of their relationships after the user
supplies a text.

## Pipeline

The project pipeline is envisaged as follows:

1. Generate a list of characters from the source text that will serve as nodes
in a character graph.
2. Identify the edges linking nodes together. These edges should represent significant
relationships between characters.
3. Characterize the nature of the edges linking nodes. Currently this is expected
to take the form of a simple characterization of relationships as positive or negative.

## Current status

Currently the project is being developed using la comtesse de Ségur's novel
for children *Le Général Dourakine*. Ultimately the goal is to be able to generalize
it to any text.

Currently the only part of the pipeline that is functional is step 2 (identification
of important relationships). This file will be updated as progress is made!
