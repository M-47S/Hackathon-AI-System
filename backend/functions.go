package main

func GetAIAnwser(json_data UserRequest) (AIResponseForm, error){
	var ai_request RequestToAI
	ai_request.session_id = json_data.session_id
	ai_request.request_id = json_data.request_id
	ai_request.message = json_data.message
	//ai_request.history = db.where(...)
	//ai.request.customer_profile = ...
	var ai_response AIResponseForm
	_, err := globalAIClient.R().
				SetBody(ai_request).
				SetResult(&ai_response).
				Post("http://localhost/v1/chat")
	if err != nil {
		return AIResponseForm{}, err
	}
	return ai_response, nil
}