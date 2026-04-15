uv pip install -e "./ArcticInference-internal[server]"
uv pip install -e ./dss-client 
uv pip install -e ./ArcticTraining-dss 
cd arctic-verl
/code/shared/verl_snowrlhf/install-h200.sh
