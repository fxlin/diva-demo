# diva-fork

Quick Start

```{shell}
# activate ssh-agent for github access
eval "$(ssh-agent -s)"
ssh-add <SSH_KEY_PATH> # add ssh key into ssh-agent

docker stop mypgdb && docker rm mypgdb && make run-postgres

cd <PATH_TO_DIVA_FORK_FOLDER>

# optional, initialize db
git pull swh demo && make build-cloud && make init-postgres
```
'''{shell}



```{shell}
#  how to run web server: python -m web/web_server
#  output folder: ./web/static/output/
```
