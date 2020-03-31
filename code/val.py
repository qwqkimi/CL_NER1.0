import bilsm_crf_model
import process_data
import numpy as np
import config
model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
#predict_text ='曾因犯故意伤害罪于2008年4月15日被北京市大兴区人⺠法院判处有期徒刑七个月;'
predict_text ='现因涉嫌犯强迫交易罪于2018年12月18日被羁押,当日被刑事拘留,2019年1月11日被逮捕,又因涉嫌犯单位行贿罪于2019年3月5日被北京市朝阳区监察委员会立案调查,现羁押于北京市朝阳区看守所。'
#predict_text = '被告人弓×,男,36岁(1977年12月12日出生)。因涉嫌犯危险驾驶罪,于2014年2月4日被北京市延庆县公安局羁押,次日被刑事拘留,现羁押于北京市延。'
str, length = process_data.process_data(predict_text, vocab)
model.load_weights('model/crf_'+config.model_name+'.h5')
raw = model.predict(str)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

cl=''

for s, t in zip(predict_text, result_tags):
    if t in ('B-CL', 'I-CL'):
        cl += ' ' + s if (t == 'B-CL') else s

print(['CL:' + cl])
