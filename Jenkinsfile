pipeline {
    agent any
    stages {
        stage('Build Model'){
            steps {
                sh 'chmod +x scripts/build.sh'
                sh './scripts/build.sh'
            }
        }
        stage('Get BEst Model'){
            steps {
                sh 'chmod +x scripts/get_best_model.sh'
                sh './scripts/get_best_model.sh'
            }
        }
    }
}