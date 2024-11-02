import json
import uuid
import os

from asgiref.sync import async_to_sync, sync_to_async
from django.http import HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from AITutor_Backend.src.BackendUtils.tutor_agents import LRUAgentCache
from AITutor_Backend.models import DatabaseManager

agent_cache_glob = LRUAgentCache(os.environ.get("AGENT_CACHE_SIZE", 32))


def make_error_response(error_msg, sid, status=405):
    return JsonResponse(
        json.dumps(
            {"session_key": f"{sid}", "success": False, "resonse": {"error": error_msg}}
        ),
        status=status,
    )


def make_environment_response(environment_data, current_state, sid, status=200):
    def __get_response_obj(current_state, environment_data):
        if current_state == 0:
            return (True, {"prompt": environment_data})
        if current_state == 1:
            return (True, {"teaching": environment_data})
        if current_state == 2:
            return (True, {"testing": environment_data})
        if current_state == 3:
            return (True, {"metrics": environment_data})
        if current_state == 4:
            return (True, {"prompt": environment_data})
        return (False, "Invalid state has occurred. Try restarting the session.")

    success, response_obj = __get_response_obj(current_state, environment_data)
    if success:
        return JsonResponse(
            {
                "session_key": sid,
                "success": success,
                "current_state": current_state,
                "response": response_obj,
            },
            status=status,
        )
    else:
        return make_error_response(response_obj, sid, status=405)


def process_session_data(data):
    # Extract User Input:
    if not "user_prompt" in data:
        return make_error_response(
            "No user prompt provided, ensure you are sending over the proper data.",
            data["session_key"],
        )
    # Further processing:
    tutor_agent = None
    session_id = data.get("session_key")

    # Check Cache for Agent:
    if session_id:
        tutor_agent = agent_cache_glob.get_agent(session_id)

    # Obtain the Agent from DB, handle Cache insertion:
    if tutor_agent is None:
        session_id, db_manager = DatabaseManager.get_tutor_env(session_id)
        tutor_agent = db_manager.load_tutor_env()

        # Add new Agent to the Cache:
        old_agent_node = agent_cache_glob.add_agent(session_id, tutor_agent)
        if old_agent_node:  # Save data back to DB if removal
            old_agent_db_manager = DatabaseManager(old_agent_node.uuid)
            old_agent_db_manager.tutor_env = old_agent_node.agent
            old_agent_db_manager.save_tutor_env()

    environment_data, current_state = tutor_agent.step(data)

    # Return Env Response:
    return make_environment_response(
        environment_data, current_state, data["session_key"]
    )


def create_and_process_session_data(data):
    # Handle the case where no session key is provided
    session_key, db_manager = DatabaseManager.create_tutor_session()
    db_manager.load_tutor_env()
    environment_data, current_state = db_manager.process_tutor_env(data)
    # Save Enviornment State:
    db_manager.save_tutor_env()
    # Return Env Response:
    return make_environment_response(environment_data, current_state, session_key)


# TODO: Fix CSRF error
@csrf_exempt
def session_view(request):
    if request.method == "GET":
        return JsonResponse({"msg": "42."})
    if request.method == "POST":
        raw_data = request.body.decode("utf-8")
        json_data = json.loads(raw_data)
        data = dict(json_data)

        data["session_key"] = json_data.get("session_key", "")
        data["is_audio"] = json_data.get("is_audio", False)
        data["user_prompt"] = json_data.get("user_prompt", "")
        # TODO: add more modalities e.g. files, audio, ...
        # Handle the case where a session key is provided:
        if "session_key" in data and data["session_key"]:
            return process_session_data(data)
        else:
            return create_and_process_session_data(data)
    else:
        return HttpResponseBadRequest("Invalid method")


# TODO: Fix CSRF error
@csrf_exempt
def load_chat_view(request):  # TODO: FIX
    if request.method == "GET":
        return JsonResponse({"msg": "42."})
    if request.method == "POST":
        raw_data = request.body.decode("utf-8")
        json_data = json.loads(raw_data)
        data = dict(json_data)
        # Handle the case where a session key is provided:
        if "session_key" in data and data["session_key"]:
            return process_session_data(data)
        else:
            return create_and_process_session_data(data)
    else:
        return HttpResponseBadRequest("Invalid method")
