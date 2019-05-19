from django.shortcuts import render
import pickle


# Create your views here.
def demo_view(request):
     content = request.POST.get('Content', None)
     if content:
          with open('./model/model.pickle', 'rb') as f:
               classifier = pickle.load(f)
          f.close()

          with open('./model/dictionary.pickle', 'rb') as f:
               dictionary = pickle.load(f)
          f.close()

          dt = dictionary.token2id
          length = len(dt)
          wids = [0] * length

          for wid in dictionary.doc2idx(content.split(' ')):
               wids[wid] += 1

          prediction = classifier.predict([wids])
          prediction = str(prediction)
          prediction = prediction[2:-2]
          return render(request,'index.html', {'error_msg': 0, 'prediction':prediction})

     else:
          error_msg = 'Please input contents'
          return render(request,'index.html', {'error_msg': error_msg, 'prediction': None})