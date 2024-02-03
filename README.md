# Zilo Technical Task

## Introduction

Below is a task definition, similar to something you may work on while at Zilo. We have provided a dataset and some guidance on what we are looking for, but the rest is up to you. We are looking for a solution that is well designed, easy to understand, and production ready.

We emphasize the importance of production-grade quality. We are eager to observe your proficiency in navigating the entire ML pipeline. A useful guideline to follow would be to ensure your work is of a standard that you'd feel confident submitting as a pull request to a GitHub repository.

In any later interview, we may ask you to explain your reasoning behind certain decisions, so please make sure you use a notebook or similar to document your thought process. We are not just interested in the final outcome, but your thought process along the way.

Python is the suggested language for this task.

### Notes

- You may use any resource to help you complete the task, we want to see how you naturally work.

- Please only use the data provided here, do not combine with any external data.

- Everyone will try things and not have them succeed, make a note of what you tried for us and why they didn’t work.

- Although part of the task is to create a working model, we are considering aspects alongside model quality.

- If you are unsure of anything, don’t hesitate to get in touch - we’re not trying to catch you out anywhere, we want to make sure everything is clear!

- If you feel like you are spending too much time on the task, we are happy to accept partial solutions.

- [Optional] Conversely, if you find yourself with more time on your hands - show us how you would extend this task.

### Estimated Time:
The estimated time to complete this task will depend on your familiarity with the technologies and techniques involved, we believe it will take roughly 2-4 hours. It is recommended to spend a reasonable amount of time on each aspect of the task, but there is no strict time limit. Focus on demonstrating your understanding of the requirements and delivering a well-designed and functional solution.

### Submission:
Please submit your solution, including the source code, any necessary configuration files, and documentation. Provide clear instructions on how to run and test the API locally.

Feel free to reach out if you have any questions or need further clarification on any aspect of the task. Good luck!

## Task Definition

### Problem Statement

Develop a software solution that can effectively identify and classify SMS messages as either 'spam' or 'not spam'. This involves creating a machine learning model or algorithm that can analyze the content of an SMS message and make a prediction based on its understanding of what constitutes spam. The solution should be able to learn from a dataset of SMS messages that have been pre-classified as 'spam' or 'not spam', and apply this learning to new, unseen messages. The ultimate goal is to reduce the amount of spam messages that reach users' inboxes by accurately identifying and filtering out spam.

### The Data

The data is a set of email subjects and a binary spam classification. We will withhold some data for validation of your model.

Columns included are as follows:

- Label - `spam`/`ham`

- Text - Content of the email subject

### Acceptance Criteria

- The API must incorporate an endpoint that accepts data from an SMS message and subsequently returns a response that includes a spam classification.

- The solution design must be flexible enough to accommodate easy switching of models, and the addition of new models in the future.

- The solution must be able to run on any machine, locally or in the cloud.

- The solution must be documented and easy to understand.
