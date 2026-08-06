#!/usr/bin/env bash
# Activates the cram-env virtualenv and starts the cram-viz server.

source /etc/profile.d/virtualenvwrapper.sh
workon cram-viz
cram-viz
