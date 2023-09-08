# WebApp for Online Absorption Prediction 

This web app is not deployed on AWS anymore, becasue I do not have funding to support it.

This github repo has all you need to run the webapp except the two machine learning models as they are too big for github to store. Check the [official keyhole webpage](https://rubyjiang18.github.io/keyholeofficial/) to download the trained models, one for ConvNeXt-T deep learning model and another for the Random Forest model.

## Instructions to run the website locally without Docker
1. Create a new environemnt named webapp `conda create --name webapp`
2. Activate this environment `conda activate webapp`
3. Install all dependencies `pip install -r requirements.txt`
4. Run the flask app by using this command: `flask run`
5. If you need to specify the specific Python file containing your Flask app, you can use the FLASK_APP environment variable. For example, `export FLASK_APP=app.py`
6. Use `sudo lsof -i :<port_number>` to check the running instances
7. To kill the current running Flask app, `CTRL + C` in termal or `kill <pid>`
### To deploy this web app without Docker to AWS EC2 instance
You can follow [this tutorial](https://www.twilio.com/blog/deploy-flask-python-app-aws).

## Instructions to run the website locally using Docker
1. [Install](https://docs.docker.com/get-docker/) and Start docker on your machine, either the docker application on Mac/Windows.
2. Run Docker `docker-compose up`<br/>
Now you can check the web app running at http://localhost:7004
### To deploy this Docker app to AWS ECS
I followed [this tutorial](https://cto.ai/blog/deploying-a-docker-application-to-aws-ecs/)



