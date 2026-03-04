import requests
import json
import os
from github import Github
from github import Auth as auth
from dotenv import load_dotenv
load_dotenv()
Organization="AOSSIE-Org"
token=os.getenv("PAT_token")
authenticate=auth.Token(token)
#create a instance
g=Github(auth=authenticate)
ORG="AOSSIE-Org"
Organization=g.get_organization(ORG)
# fetch all repositories
repositories=Organization.get_repos()
repo=[]
for i in repositories:
    repo.append(i)
pic=Organization.get_repo("PictoPy")
workflows=[]
for x in pic.get_workflows():
    workflows.append(x.name)
print(workflows)

