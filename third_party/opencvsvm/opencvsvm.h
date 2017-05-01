#ifndef OPENCVSVM_H
#define	OPENCVSVM_H

#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <vector>
#include <locale.h>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

// Precision to use (float / double)
typedef float prec;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class OpenCVSVM {
private:
    char* line_;
    int max_line_len_;

    cv::Mat labels_;
    cv::Mat features_;
    cv::Ptr<cv::ml::SVM> svm_;

    void exit_input_error(int line_num) {
        fprintf(stderr, "Wrong input format at line %d\n", line_num);
        exit(1);
    }

    OpenCVSVM() {
        line_ = NULL;
        max_line_len_ = 262144; // WARNING: read_problem() won't work if line is longer

        svm_ = cv::ml::SVM::create();
        svm_->setType(cv::ml::SVM::C_SVC);
        svm_->setKernel(cv::ml::SVM::LINEAR);
        svm_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
/*        predict_probability = 1; // 0 or 1
        // Init some parameters
        param.cache_size = 512; // in MB
        param.coef0 = 0.0; // for poly/sigmoid kernel
        param.degree = 3; // for poly kernel
        /// @WARNING eps_criteria = 0.001, eps = 0.1 which one to use?
        param.eps = 1e-3; // 0.001 stopping criteria
        param.gamma = 0; // for poly/rbf/sigmoid
        param.kernel_type = 0; // libsvm::LINEAR; // type of kernel to use
        param.nr_weight = 0; // for C_SVC
        param.nu = 0.5; // for NU_SVC, ONE_CLASS, and NU_SVR
        param.p = 0.1; // for EPSILON_SVR, epsilon in loss function?
        param.probability = predict_probability; // do probability estimates
        param.C = 0.01; // From paper, soft classifier
        param.shrinking = 0; // use the shrinking heuristics, equals flag -h
        param.svm_type = EPSILON_SVR; // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
        param.weight_label = NULL; // for C_SVC
        param.weight = NULL; // for C_SVC
        model = NULL;
        max_line_len_ = 1024;
        x_space = NULL;
*/
    }

    /**
     * Free the used pointers
     */
    virtual ~OpenCVSVM() {
    }

public:

    static OpenCVSVM* getInstance();

    const char* getSVMName() const {
        return "OpenCVSVM";
    }
    
    /**
     * Reads in a file in svmlight format
     * @param filename
     */
    void read_problem(char *filename) {
        /// @WARNING: This is really important, ROS seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
        // Do not use the system locale setlocale(LC_ALL, "C");
        setlocale(LC_NUMERIC,"C");
        setlocale(LC_ALL, "POSIX");
//        cout.getloc().decimal_point ();
        int elements, max_index, inst_max_index, i, j;
        FILE *fp = fopen(filename, "r");
        char *endptr;
        char *idx, *val, *label;

        if (fp == NULL) {
            fprintf(stderr, "can't open input file %s\n", filename);
            exit(1);
        }

        int lines_count = 0;
        int features_count = 0;

        line_ = Malloc(char, max_line_len_);
        while (fgets(line_, max_line_len_, fp) != NULL) {
            char *p = strtok(line_, " \t"); // label
            // features
            while (1) {
                p = strtok(NULL, " \t");
                if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                    break;
                ++features_count;
            }
            ++lines_count;
        }
        rewind(fp);

        printf("rewind! features: %d, lines: %d\n", features_count, lines_count);
        const int features_per_vec = features_count/lines_count;
        labels_.create(lines_count, 1, CV_32SC1);
        features_.create(lines_count, features_per_vec, CV_32FC1);

        int *labels_data = labels_.ptr<int>();
        float *features_data = features_.ptr<float>();
        int index = -1;
        max_index = 0;
        for (i = 0; i < lines_count; i++) {
            inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
            fgets(line_, max_line_len_, fp);
            label = strtok(line_, " \t\n");
            if (label == NULL) { // empty line
                printf("Empty line encountered!\n");
                exit_input_error(i + 1);
            }

            labels_data[i] = strtol(label, &endptr, 10);
            if (endptr == label || *endptr != '\0') {
                printf("Wrong line ending encountered!\n");
                exit_input_error(i + 1);
            }

            j = 0;
            while (1) {
                idx = strtok(NULL, ":");
                val = strtok(NULL, " \t");

                if (val == NULL)
                    break;

                errno = 0;
                index = (int) strtol(idx, &endptr, 10);
                if (endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index) {
                    printf("File input error at feature index encountered: %i, %i, %i, %i!\n", endptr == idx, errno != 0, *endptr != '\0', index <= inst_max_index);
                    exit_input_error(i + 1);                    
                } else {
                    inst_max_index = index;
                }
                errno = 0;
                features_data[i*features_per_vec + j] = std::strtod(val, &endptr);
//                printf("Value: '%f'\n", x_space[j].value);
                if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
                    printf("File input error at feature value encountered: %i, %i, %i: '%s'\n",endptr == val, errno != 0, (*endptr != '\0' && !isspace(*endptr)), endptr);
                    exit_input_error(i + 1);
                }
                ++j;
            }

            if (inst_max_index > max_index)
                max_index = inst_max_index;
            index = -1;
        }
/*
        if (param.gamma == 0 && max_index > 0) {
            param.gamma = 1.0 / max_index;
        }
*/
/*
        if (param.kernel_type == PRECOMPUTED) {
            for (i = 0; i < prob.l; i++) {
                if (prob.x[i][0].index != 0) {
                    fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
                    exit(1);
                }
                if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
                    fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
                    exit(1);
                }
            }
        }
*/
        fclose(fp);
        printf("READ FEATURES!!");
    }

    /**
     * Modelfile is saved to filesystem and contains the svm parameters used for training, which need to be retrieved for classification
     * Only makes sense after training was done
     * @param _modelFileName file name to save the model to
     */
    void saveModelToFile(const std::string _modelFileName) {
        svm_->save(_modelFileName);
    }

    /**
     * After read in the training samples from a file, set parameters for training and call training procedure
     */
    void train() {
        printf("features_: %dx%d, labels_: %dx%d", features_.rows, features_.cols, labels_.rows, labels_.cols);
        cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(features_, cv::ml::ROW_SAMPLE, labels_);
        svm_->trainAuto(data);
    }

    /**
     * Generates a single detecting feature vector (vec1) from the trained support vectors, for use e.g. with the HOG algorithm
     * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
     * @param detector resulting single detector vector for use in openCV HOG
     * @param detector_indices vector containing indices of features inside detector
     */
    void getSingleDetectingVector(std::vector<prec>& detector, std::vector<unsigned int>& detector_indices) {
        // Now we use the trained svm to retrieve the single detector vector
        printf("Calculating single detecting feature vector out of support vectors (may take some time)\n");
        detector.clear();
        detector_indices.clear();


        std::vector<double> alphas;
        std::vector<int> svidx;
        double rho = svm_->getDecisionFunction(0, alphas, svidx);
        printf("RHO: %lf\n", rho);

        cv::Mat svecs = svm_->getSupportVectors();
        printf("Total number of support vectors: %d \n", svecs.rows);
        // Walk over every support vector and build a single vector
        for (int ssv = 0; ssv < svecs.rows; ++ssv) { // Walks over available classes (e.g. +1, -1 representing positive and negative training samples)
            // Retrieve the current support vector from the training set
            cv::Mat support_vec = svecs.row(ssv); // Get next support vector ssv==class, 2nd index is the component of the SV
            const float *support_vec_data = support_vec.ptr<float>();
            // sv_coef[i] = alpha[i]*sign(label[i]) = alpha[i] * y[i], where i is the training instance, y[i] in [+1,-1]
            for (int i = 0; i < support_vec.cols; ++i) { // index=-1 indicates the end of the array
                if (ssv == 0) { // During first loop run determine the length of the support vectors and adjust the required vector size
                    detector.push_back(support_vec_data[i] * alphas[i]);
                    // Assume indices are in order
                    detector_indices.push_back(svidx[i]); // Holds the indices for the corresponding values in detector, mapping from singleVectorComponent to support_vec[singleVectorComponent].index!
                } else {
                    if (i > detector.size()) { // Catch oversized vectors (maybe from differently sized images?)
                        printf("Warning: Component %d out of range, should have the same size as other/first vector\n", i);
                    } else
                        detector.at(i) += (support_vec_data[i] * alphas[i]);
                }
            }
        }

        // This is a threshold value which is also recorded in the lear code in lib/windetect.cpp at line 1297 as linearbias and in the original paper as constant epsilon, but no comment on how it is generated
        //detector.push_back(-rho); // Add threshold
        //detector_indices.push_back(-1); // Add maximum unsigned int as index indicating the end of the vector
        //detector_indices->push_back(UINT_MAX); // Add maximum unsigned int as index indicating the end of the vector
    }

    /**
     * Return model detection threshold / bias
     * @return detection threshold / bias
     */
    double getThreshold() const {
        return 0.2f;
    }

};

/// Singleton, @see http://oette.wordpress.com/2009/09/11/singletons-richtig-verwenden/
OpenCVSVM* OpenCVSVM::getInstance() {
    static OpenCVSVM theInstance;
    return &theInstance;
}

#endif	/* OPENCVSVM_H */
