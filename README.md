# diva-fork

Quick Start

```{shell}
# activate ssh-agent for github access
eval "$(ssh-agent -s)"
ssh-add <SSH_KEY_PATH> # add ssh key into ssh-agent

# Init DB
docker stop mypgdb && docker rm mypgdb && make run-postgres && sleep 10 && make init-postgres && make fixture-postgres


make setup-env
make run-yolo
make run-cloud

```

## WEB

```{shell}
# Run web server: python -m web.web_server
# Output directory: ./web/static/output
```
