# Retips

To run the code in this repo, you'll need to get a python environment capable of doing that. The two sections below are designed to walk you through the process of creating such a python environment inside an Apptainer, and then running Jupyter Lab inside that Apptainer.

## Creating Apptainer environment
The following steps will result in the creation of an apptainer `.sif` file which will include a python environment capable of running Hugging Face and OpenAI models, as well as JupyterLab. **You will only need to follow these steps one time**, and then you can use the resulting `.sif` file as often as you like.

1. Start an interactive JupyterLab session.
2. Open a terminal.
3. Navigate to wherever this RETIPS repo is located in your drive structure.
4. Navigate to the subdirectory `environment_apptainer`. (If you don't have this, you need to `git pull`.)
5. Run the command `apptainer build llms_env.sif llms_env.def`. This will use the `.def` file that I wrote to build a `.sif` file, which will be the Apptainer container. (If you like, you can take a look at the `.def` file first -- it's written in plaintext and is pretty short.) Building the `.sif` file will take a while -- maybe 10 or 15 minutes.
6. Verify that the `.sif` file was created successfully by opening a shell inside it. To do this, run `apptainer shell llms_env.sif`. From there, you can run whatever commands you like (`python`, etc), but you should also run `pip show transformers openai torch` to verify that these three libraries are installed in the base python environment inside the container.
7. Exit the Apptainer shell using the command `exit`.
8. Back outside of the Apptainer shell, still in the `environment_apptainer` folder, run the command `echo "export RETIPS_ENV_APPTAINER_PATH=$(pwd)" >> ~/.bashrc`. This will add a new environment variable to your bash shell, which will be convenient for us down the road. Now, it will be easy for a single line of code to navigate each of us to where our `.sif` file is, even though our repositories might be in very different locations in our drive structures.

You now have an Apptainer that can run LLMs, including in JupyterLab. But! Actually starting a JupyterLab requires following the steps outlined in the section below.

## Starting a containerized Apptainer Jupyter Lab session
You may have noticed that Open OnDemand (OOD) has an option for creating a containerized JupyterLab session. Fantastic, right? Except that CCIT hasn't yet made it possible to bind your `scratch` folder into the container. That's a huge problem for us, because our giant LLMs need to live in `scratch`. So, we can't use that fantastic OOD option. We need to use a hacky workaround, described below. **You will need to follow these steps each time you want to launch a JupyterLab session.** You will need to have already followed the steps above to create a `.sif` file.

1. Log in to the Palmetto cluster through the SSH command: `ssh -L 8001:localhost:8001 {username}@login.palmetto.clemson.edu` on the command line on your local machine. This command forwards traffic from port 8001 on your local machine to port 8001 on the main login node. Replace `{username}` with your Palmetto username.
2. Run `cd $RETIPS_ENV_APPTAINER_PATH` to navigate to where your `.sif` file is.
3. Run `qsub launch_jupyterlab.pbs`. This submits the `.pbs` script I wrote to be run on the Cluster. You can read the `.pbs` script first if you like. Palmetto documentation on `qsub` is [here](https://docs.rcd.clemson.edu/palmetto/jobs/submit/), and is quite helpful.
4. Once the job actually starts running (which may take time, especially if the Cluster is busy), it will write two files to the folder where your `.sif` folder is: `jupyter_node.txt` and `jupyter_url.txt`. You'll need the contents of each of these two text files to connect to the Jupyter Lab session. To get them, run `cat jupyter_*`.
5. Once you have the node and token, run `ssh -L 127.0.0.1:8001:localhost:8888 {node}`, replacing `{node}` with the node you copied earlier. This creates another SSH tunnel to the compute node where your Jupyter Lab session is running (since it is not running on the login node).
6. In your browser on your local machine, go to the url that you found in `jupyter_url.txt`.

You should now be able to run Jupyter Lab in your browser, and you will be running everything inside your Apptainer! Let me know if you run into problems.
