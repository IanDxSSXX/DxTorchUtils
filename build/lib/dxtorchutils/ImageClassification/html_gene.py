import cv2
from jinja2 import Environment, FileSystemLoader
import os
import logging
import torch

DIR_PATH: str
METRICS: [str]


def home_template_gene():
    """
    Generate the home template html in the given directory path
    :return:
    """
    html = """<meta http-equiv="Content-Type"content="text/html;charset=utf-8">
<html align="left">
<h1> {{ paras.title }} </h1>
<body>

    <div>
    <h2> Description </h2>
    {{ paras.description }}
    </div>

    <div style="overflow: hidden">
    <h2> Dataset </h2>
    <table border="1" cellspacing="5" cellpadding="5" align="left">
    <tr>
        <th> training </th>
        <th> validation </th>
        <th> testing </th>
    </tr>
    <tr>
        <td> {{ paras.training_num }} </th>
        <td> {{ paras.validation_num }} </th>
        <td> {{ paras.testing_num }}</th>
    </tr>
    </table>
    </div>

    <div style="overflow: hidden">
    <h2> Resource </h2>
    <table border="1" cellspacing="5" cellpadding="5" align="left">
    <tr>
        <th> model name </th>
        <th> original paper </th>
        <th> source code </th>
    </tr>
    {% for model in paras.models %}
        <tr>
            <td> {{ model.model_name }} </td>
            <td> <a href={{ model.paper_url }}> {{ model.paper_name }} </a> </td>
            <td> <a href={{ model.code_url }}> {{ model.code_url }} </a> </td>
        </tr>
    {% endfor %}
    </table>
    </div>

    <div style="overflow: hidden">
        <h2> Results </h2>
        <table border="1" cellspacing="5" cellpadding="5" align="left">
        <tr>
            <th> model name </th>
            {% for metric in paras.metrics %}
                <th> {{ metric }} </th>
            {% endfor %}
            <th> details </th>
        </tr>
        {% for model in paras.models %}
            <tr>
                <td> {{ model.model_name }} </td>
                {% for metric_value in model.metrics_value %}
                    <td> {{ metric_value }} </td>
                {% endfor %}
                <td> <a href="result_detail_{{ model.model_name_ul }}.html"> details </a> </td>
            </tr>
        {% endfor %}
        </table>
    </div>
</body>
</html>
"""

    with open("{}/template_home.html".format(DIR_PATH), "w") as f:
        f.write(html)


def detail_template_gene():
    """
    Generate the detail template html in the given directory path
    :return:
    """
    html = """<meta http-equiv="Content-Type"content="text/html;charset=utf-8">
<html align="left">
<h1> {{ model.model_name }} </h1>
<body>
    <table border="1" cellspacing="5" cellpadding="5" align="left">
    <tr>
        <th>id</th>
        <th>raw</th>
        <th>label</th>
        <th>prediction</th>
        {% for metric in model.metrics %}
            <th> {{ metric }} </td>
        {% endfor %}
    </tr>

    {% for image in model.images %}
    <tr align="center">
        <td> {{ image.img_id }} </td>
        <td> <img src={{ image.raw_path }} height="128"> </td>
        <td> {{ image.label }} </td>
        <td> {{ image.prediction }} </td>
        {% for metric_value in image.metrics_value %}
            <td> {{ metric_value }} </td>
        {% endfor %}
    </tr>
    {% endfor %}
    </table>
</body>
</html>
"""
    with open("{}/template_detail.html".format(DIR_PATH), "w") as f:
        f.write(html)


class ImageParas:
    def __init__(
            self,
            img_id: int,
            raw: torch.FloatTensor,
            label: int,
            prediction: int,
            metrics_value: [float]
    ):
        self.img_id = img_id
        self.raw = raw
        self.label = label
        self.prediction = prediction

        self.raw_path = ""
        self.label_path = ""
        self.prediction_path = ""

        self.metrics_value = []
        for metric_value in metrics_value:
            self.metrics_value.append("{:.4f}".format(metric_value))


class ModelParas:
    def __init__(
            self,
            model_name: str = "Add model name",
            paper_url: str = "Add paper url",
            paper_name: str = "Add paper name",
            code_url: str = "Add code url",
            images=None
    ):
        if images is None:
            images = []

        self.model_name = model_name
        self.model_name_ul = model_name.replace(" ", "_")
        self.paper_url = paper_url
        self.paper_name = paper_name
        self.code_url = code_url
        self.metrics_value = ["0"] * len(METRICS)
        self.metrics = METRICS
        self.images = images

    def add_image(self, image: ImageParas):
        self.images.append(image)

        for idx in range(len(self.metrics_value)):
            self.metrics_value[idx] \
                = "{:.4f}".format(
                ((float(self.metrics_value[idx]) * (len(self.images) - 1) +
                  float(self.images[-1].metrics_value[idx])) / len(self.images)))

        if not os.path.exists("{}/images/raw".format(DIR_PATH)):
            os.makedirs("{}/images/raw".format(DIR_PATH))

        raw_path = "{}/images/raw/{:05}.png".format(DIR_PATH, image.img_id)

        if not os.path.exists(raw_path):
            cv2.imwrite(raw_path, image.raw)

        image.raw_path = "images/raw/{:05}.png".format(image.img_id)

    def add_images(self, images: [ImageParas]):
        for image in images:
            self.add_image(image)


class RenderParas:
    def __init__(
            self,
            dir_path="results",
            title: str = "Set a title",
            description: str = "Set a description",
            training_num: int = 0,
            validation_num: int = 0,
            testing_num: int = 0,
            metrics=None,
    ):
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "AUC"]

        global METRICS
        METRICS = metrics

        global DIR_PATH
        DIR_PATH = dir_path

        self.title = title
        self.description = description
        self.training_num = training_num
        self.validation_num = validation_num
        self.testing_num = testing_num
        self.metrics = METRICS
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def gene_html(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)

        if not os.path.exists(DIR_PATH):
            os.makedirs(DIR_PATH)

        home_template_gene()
        detail_template_gene()

        env = Environment(loader=FileSystemLoader(DIR_PATH))
        template = env.get_template("template_home.html")
        with open("{}/result_home.html".format(DIR_PATH), "w") as f:
            html_rendered = template.render(paras=self)
            f.write(html_rendered)

        for model in self.models:
            detail_template = env.get_template("template_detail.html")
            with open("{}/result_detail_{}.html".format(DIR_PATH, model.model_name_ul), "w") as f:
                html_detail_rendered = detail_template.render(model=model)
                f.write(html_detail_rendered)

        os.remove("{}/template_home.html".format(DIR_PATH))
        os.remove("{}/template_detail.html".format(DIR_PATH))

        logger.info("Result html home is generated at {}/result_home.html".format(DIR_PATH))
