# WebApp for Online Absorption Prediction 

This web app has been deployed on AWS, check this url: http://3.80.140.107:8080/

Instructions to run the website locally
1. Create a new environemnt named webapp `conda create --name webapp`
2. Activate this environment `conda activate webapp`
3. Install all dependencies `pip install -r requirements.txt`
4. Run the flask app by using this command: `flask run`
5. If you need to specify the specific Python file containing your Flask app, you can use the FLASK_APP environment variable. For example, `export FLASK_APP=app.py`
6. Use `sudo lsof -i :<port_number>` to check the running instances
7. To kill the current running Flask app, `CTRL + C` in termal or `kill <pid>`

To deploy this web app to AWS EC2 instance, I followed [this tutorial](https://www.twilio.com/blog/deploy-flask-python-app-aws).