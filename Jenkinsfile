pipeline {
    agent any
    stages {
        stage('Build Model'){
            steps {
                sh 'chmod 777 scripts/build.sh'
                sh './scripts/build.sh'
            }
        }
        stage('Get BEst Model'){
            steps {
                sh 'chmod 777 scripts/get_best_model.sh'
                sh './scripts/get_best_model.sh'
            }
        }
    }
}