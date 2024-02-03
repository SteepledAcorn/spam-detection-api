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
