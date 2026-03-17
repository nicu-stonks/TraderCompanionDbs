import json
import requests
from copy import deepcopy
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import timedelta
from .models import LLMConfig, Supplement, DailyRecord, FoodItem
from .serializers import LLMConfigSerializer, SupplementSerializer, DailyRecordSerializer, FoodItemSerializer
import re


def _build_state_snapshot(record):
    return {
        "foods_eaten": record.foods_eaten,
        "supplements_taken": deepcopy(record.supplements_taken) if isinstance(record.supplements_taken, list) else record.supplements_taken,
        "nutrient_completion": deepcopy(record.nutrient_completion) if isinstance(record.nutrient_completion, dict) else record.nutrient_completion,
        "nutrient_sources": deepcopy(record.nutrient_sources) if isinstance(record.nutrient_sources, dict) else record.nutrient_sources,
        "recommendations": record.recommendations,
    }


def _apply_state_snapshot(record, snapshot):
    record.foods_eaten = snapshot.get("foods_eaten", "")
    record.supplements_taken = snapshot.get("supplements_taken", [])
    record.nutrient_completion = snapshot.get("nutrient_completion", {})
    record.nutrient_sources = snapshot.get("nutrient_sources", {})
    record.recommendations = snapshot.get("recommendations", "")


def _latest_snapshot_from_history(history):
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "model":
            snapshot = msg.get("state_snapshot")
            if isinstance(snapshot, dict):
                return snapshot
    return None

class LLMConfigView(APIView):
    def get(self, request):
        config, _ = LLMConfig.objects.get_or_create(pk=1)
        serializer = LLMConfigSerializer(config)
        return Response(serializer.data)

    def post(self, request):
        config, _ = LLMConfig.objects.get_or_create(pk=1)
        serializer = LLMConfigSerializer(config, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SupplementListCreateView(APIView):
    def get(self, request):
        supplements = Supplement.objects.all()
        serializer = SupplementSerializer(supplements, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = SupplementSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SupplementDeleteView(APIView):
    def delete(self, request, pk):
        try:
            supp = Supplement.objects.get(pk=pk)
            supp.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Supplement.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class FoodItemListCreateView(APIView):
    def get(self, request):
        foods = FoodItem.objects.all()
        serializer = FoodItemSerializer(foods, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = FoodItemSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FoodItemDeleteView(APIView):
    def delete(self, request, pk):
        try:
            food = FoodItem.objects.get(pk=pk)
            food.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except FoodItem.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class DailyRecordListView(APIView):
    def get(self, request):
        records = DailyRecord.objects.order_by('-date')
        serializer = DailyRecordSerializer(records, many=True)
        return Response(serializer.data)

class DailyRecordDetailView(APIView):
    def get(self, request, date):
        try:
            record = DailyRecord.objects.get(date=date)
            serializer = DailyRecordSerializer(record)
            return Response(serializer.data)
        except DailyRecord.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, date):
        try:
            record = DailyRecord.objects.get(date=date)
            record.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except DailyRecord.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

class ChatView(APIView):
    def post(self, request, date):
        prompt = request.data.get('prompt', '')
        selected_model = request.data.get('model_name', '')

        config = LLMConfig.objects.first()
        if not config or not config.api_key:
            return Response({"error": "No API Key configured. Please configure the LLM first."}, status=400)

        chosen_model = selected_model if selected_model else config.model_name
        record, _ = DailyRecord.objects.get_or_create(date=date)
        
        supplements_in_db = Supplement.objects.all()
        supplements_info = "\n".join([f"- {s.name}: {s.details}" for s in supplements_in_db])
        
        foods_in_db = FoodItem.objects.all()
        foods_info = "\n".join([f"- {f.name}: {f.details}" for f in foods_in_db])

        user_profile = config.user_profile if config.user_profile else "Not provided"

        system_instruction = f"""
About Me:
{user_profile}

My supplement inventory, which you can use to fill out some nutrients needs:
{supplements_info}

My food inventory, which you could recommend and I have quick access to them:
{foods_info}

Help me cover my macros, vitamins, and nutrients, etc needs for the day. You can recommend foods/supplements from my inventory or suggest ones i could buy from the shop to cover some of the things i lack.

foods_tracked and supplements_tracked are the foods/supplements I will consume today.

And another thing: I will paste the prompt in a application of mine and I need these statistics at the end of the prompt in my own application which I coded so I have saved data about how many nutrients I have taken each day for so I know what I need to work on:

The "progress" key is a dictionary of nutrients and the percentage of the daily recommended intake that I have consumed. "progress" can only have values between 0 and 100.
Nutrient Keys expected in "progress" (Ensure all these keys exist, even if 0):
- Macronutrients: Protein, Carbohydrates, Fiber, Fats (Total), Omega-3, Omega-6, Water
- Fat-Soluble Vitamins: Vitamin A, Vitamin D3, Vitamin E, Vitamin K1, Vitamin K2
- Water-Soluble Vitamins: B1 (Thiamin), B2 (Riboflavin), B3 (Niacin), B5 (Pantothenic Acid), B6 (Pyridoxine), B7 (Biotin), B9 (Folate), B12 (Cobalamin), Vitamin C, Choline
- Macrominerals: Calcium, Phosphorus, Magnesium, Sodium, Potassium, Chloride
- Trace Minerals: Iron, Zinc, Iodine, Selenium, Copper, Manganese
- Amino Acids: Glycine
- Phytonutrients: Lutein, Zeaxanthin, Anthocyanins, Lycopene, EGCG (Catechins)

Please consider in the "progress" fields only the foods which you put in the "foods_tracked" and "supplements_tracked" arrays(the foods_tracker and supplements_tracker arrays have as purpose telling me which foods/supplements are considered in the "progress" fields).
Respond with exactly the following format at the very end of your message so my application can recognize it correctly:
```json
{{
  "progress": {{"Protein": 80, "Vitamin C": 100, "Omega-3": 50, ...all other keys... }},
    "sources": {{"Protein": ["500g carbonara", "5 eggs"], "Vitamin C": ["oranges"], "Omega-3": ["1 omega 3 pill"], ...every non-zero nutrient MUST have an entry here...}},
    "foods_tracked": ["500g carbonara", "5 eggs"],
    "supplements_tracked": ["1 omega 3 pill", "1 choline"],
  "recommendations": "Skip Vitamin C because you ate oranges."
}}
```
Only use the ````json ... ```` formatting block once at the end of your response.
"""
        
        history = list(record.chat_history)
        
        messages_to_send = []
        for msg in history:
            # Strip out internal keys like state_snapshot before sending to Gemini API
            messages_to_send.append({
                "role": msg.get("role", "user"),
                "parts": msg.get("parts", [])
            })
            
        messages_to_send.append({"role": "user", "parts": [{"text": system_instruction + "\n\nUser Prompt: " + prompt}]})

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={config.api_key}"
        
        payload = {
            "contents": messages_to_send,
            "generationConfig": {
                "temperature": 0.5,
            }
        }
        
        try:
            resp = requests.post(api_url, json=payload, timeout=120)
            data = resp.json()
            if 'error' in data:
                return Response({"error": data['error'].get('message', 'Unknown API Error')}, status=400)
                
            try:
                model_text = data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                finish_reason = data.get('candidates', [{}])[0].get('finishReason', 'UNKNOWN')
                if finish_reason != 'STOP':
                    return Response({"error": f"Response was blocked or missing. Finish reason: {finish_reason}"}, status=400)
                return Response({"error": "Failed to parse model response.", "raw_data": data}, status=500)
            
            # Extract JSON payload
            json_block = re.search(r'```json\n(.*?)\n```', model_text, re.DOTALL)
            progress = record.nutrient_completion
            recommendations_text = record.recommendations
            
            if json_block:
                try:
                    extracted = json.loads(json_block.group(1))
                    if 'progress' in extracted:
                        progress = extracted['progress']
                    if 'recommendations' in extracted:
                        recommendations_text = extracted['recommendations']
                    if 'foods_tracked' in extracted:
                        record.foods_eaten = "\n".join([f"- {item}" for item in extracted['foods_tracked']]) if isinstance(extracted['foods_tracked'], list) else str(extracted['foods_tracked'])
                    if 'supplements_tracked' in extracted:
                        record.supplements_taken = extracted['supplements_tracked']
                    if 'sources' in extracted:
                        record.nutrient_sources = extracted['sources']
                except Exception:
                    pass
                model_text = model_text.replace(json_block.group(0), '').strip()

            # Save the new state
            record.nutrient_completion = progress
            record.recommendations = recommendations_text
            
            state_snapshot = _build_state_snapshot(record)

            history.append({"role": "user", "parts": [{"text": prompt}]}) 
            history.append({"role": "model", "parts": [{"text": model_text}], "state_snapshot": state_snapshot})
            record.chat_history = history
            
            record.save()
            
            serializer = DailyRecordSerializer(record)
            return Response(serializer.data)
            
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class ChatPromptView(APIView):
    def post(self, request, date):
        prompt = request.data.get('prompt', '')
        config = LLMConfig.objects.first()
        record, _ = DailyRecord.objects.get_or_create(date=date)
        
        supplements_in_db = Supplement.objects.all()
        supplements_info = "\n".join([f"- {s.name}: {s.details}" for s in supplements_in_db])
        
        foods_in_db = FoodItem.objects.all()
        foods_info = "\n".join([f"- {f.name}: {f.details}" for f in foods_in_db])

        user_profile = config.user_profile if config and config.user_profile else "Not provided"

        system_instruction = f"""
About Me:
{user_profile}

My supplement inventory, which you can use to fill out some nutrients needs:
{supplements_info}

My food inventory, which you could recommend and I have quick access to them:
{foods_info}

Help me cover my macros, vitamins, and nutrients, etc needs for the day. You can recommend foods/supplements from my inventory or suggest ones i could buy from the shop to cover some of the things i lack.

foods_tracked and supplements_tracked are the foods/supplements I will consume today.

And another thing: I will paste the prompt in a application of mine and I need these statistics at the end of the prompt in my own application which I coded so I have saved data about how many nutrients I have taken each day for so I know what I need to work on:

The "progress" key is a dictionary of nutrients and the percentage of the daily recommended intake that I have consumed. "progress" can only have values between 0 and 100.
Nutrient Keys expected in "progress" (Ensure all these keys exist, even if 0):
- Macronutrients: Protein, Carbohydrates, Fiber, Fats (Total), Omega-3, Omega-6, Water
- Fat-Soluble Vitamins: Vitamin A, Vitamin D3, Vitamin E, Vitamin K1, Vitamin K2
- Water-Soluble Vitamins: B1 (Thiamin), B2 (Riboflavin), B3 (Niacin), B5 (Pantothenic Acid), B6 (Pyridoxine), B7 (Biotin), B9 (Folate), B12 (Cobalamin), Vitamin C, Choline
- Macrominerals: Calcium, Phosphorus, Magnesium, Sodium, Potassium, Chloride
- Trace Minerals: Iron, Zinc, Iodine, Selenium, Copper, Manganese
- Amino Acids: Glycine
- Phytonutrients: Lutein, Zeaxanthin, Anthocyanins, Lycopene, EGCG (Catechins)

Please consider in the "progress" fields only the foods which you put in the "foods_tracked" and "supplements_tracked" arrays(the foods_tracker and supplements_tracker arrays have as purpose telling me which foods/supplements are considered in the "progress" fields).
Respond with exactly the following format at the very end of your message so my application can recognize it correctly:
```json
{{
  "progress": {{"Protein": 80, "Vitamin C": 100, "Omega-3": 50, ...all other keys... }},
    "sources": {{"Protein": ["500g carbonara", "5 eggs"], "Vitamin C": ["oranges"], "Omega-3": ["1 omega 3 pill"], ...every non-zero nutrient MUST have an entry here...}},
    "foods_tracked": ["500g carbonara", "5 eggs"],
    "supplements_tracked": ["1 omega 3 pill", "1 choline"],
  "recommendations": "Skip Vitamin C because you ate oranges."
}}
```
Only use the ````json ... ```` formatting block once at the end of your response.
"""
        
        full_text = system_instruction + "\n\n=== CONVERSATION HISTORY ===\n"
        for msg in list(record.chat_history):
            role = "AI" if msg.get("role") == "model" else "User"
            text = msg["parts"][0]["text"]
            full_text += f"{role}: {text}\n\n"
            
        full_text += f"User: {prompt}\n\n"
        full_text += "Please provide your response following the strict JSON format at the end."
        
        return Response({"full_prompt": full_text})

class ChatManualSubmitView(APIView):
    def post(self, request, date):
        prompt = request.data.get('prompt', '')
        model_text = request.data.get('model_response', '')

        record, _ = DailyRecord.objects.get_or_create(date=date)
        
        history = list(record.chat_history)
        
        json_block = re.search(r'```json\n(.*?)\n```', model_text, re.DOTALL)
        progress = record.nutrient_completion
        recommendations_text = record.recommendations
        
        if json_block:
            try:
                extracted = json.loads(json_block.group(1))
                if 'progress' in extracted:
                    progress = extracted['progress']
                if 'recommendations' in extracted:
                    recommendations_text = extracted['recommendations']
                if 'foods_tracked' in extracted:
                    record.foods_eaten = "\n".join([f"- {item}" for item in extracted['foods_tracked']]) if isinstance(extracted['foods_tracked'], list) else str(extracted['foods_tracked'])
                if 'supplements_tracked' in extracted:
                    record.supplements_taken = extracted['supplements_tracked']
                if 'sources' in extracted:
                    record.nutrient_sources = extracted['sources']
            except Exception:
                pass
            model_text = model_text.replace(json_block.group(0), '').strip()

        record.nutrient_completion = progress
        record.recommendations = recommendations_text

        state_snapshot = _build_state_snapshot(record)
        
        history.append({"role": "user", "parts": [{"text": prompt}]}) 
        history.append({"role": "model", "parts": [{"text": model_text}], "state_snapshot": state_snapshot})
        record.chat_history = history
        
        record.save()
        
        serializer = DailyRecordSerializer(record)
        return Response(serializer.data)


class ChatRollbackView(APIView):
    def post(self, request, date):
        record, _ = DailyRecord.objects.get_or_create(date=date)
        history = list(record.chat_history)

        before_message_index = request.data.get('before_message_index', None)
        discard_pairs = request.data.get('discard_pairs', None)

        if before_message_index is not None:
            try:
                before_message_index = int(before_message_index)
            except (TypeError, ValueError):
                return Response({"error": "before_message_index must be an integer."}, status=400)

            if before_message_index < 0 or before_message_index > len(history):
                return Response({"error": "before_message_index out of range."}, status=400)

            new_history = history[:before_message_index]
        else:
            try:
                discard_pairs = int(discard_pairs) if discard_pairs is not None else 1
            except (TypeError, ValueError):
                return Response({"error": "discard_pairs must be an integer."}, status=400)

            if discard_pairs < 0:
                return Response({"error": "discard_pairs must be >= 0."}, status=400)

            discard_messages = discard_pairs * 2
            if discard_messages == 0:
                new_history = history
            elif discard_messages >= len(history):
                new_history = []
            else:
                new_history = history[:-discard_messages]

        snapshot = _latest_snapshot_from_history(new_history)
        if snapshot:
            _apply_state_snapshot(record, snapshot)
        else:
            record.foods_eaten = ""
            record.supplements_taken = []
            record.nutrient_completion = {}
            record.nutrient_sources = {}
            record.recommendations = ""

        record.chat_history = new_history
        record.save()

        serializer = DailyRecordSerializer(record)
        return Response({
            "discarded_messages": len(history) - len(new_history),
            "history_length": len(new_history),
            "record": serializer.data,
        })
