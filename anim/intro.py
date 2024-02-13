import numpy as np
from manim import *

class Motivation(Scene):
    def construct(self):
        self.camera.background_color = "#282a36"
        self.wait(1)
        attention_paper = ImageMobject("./img/Attention_is_all_you_need.png").to_edge(LEFT)
        self.add(attention_paper)
        self.wait(1)
        gpt1 = SVGMobject("./img/openai.svg").scale(0.5).next_to(attention_paper).shift(1.5*UP+RIGHT).set_color(WHITE)
        gpt1_text = Text("GPT-1 (Released 2018)").scale(0.5).next_to(gpt1, RIGHT)
        self.add(gpt1)
        self.play(Write(gpt1_text))
        self.wait(1)
        gpt2 = SVGMobject("./img/openai.svg").scale(0.5).next_to(attention_paper).shift(RIGHT).set_color(WHITE)
        gpt2_text = Text("GPT-2 (Released 2019)").scale(0.5).next_to(gpt2, RIGHT)
        self.add(gpt2)
        self.play(Write(gpt2_text))
        self.wait(1)
        gpt3 = SVGMobject("./img/openai.svg").scale(0.5).next_to(attention_paper).shift(1.5*DOWN+RIGHT).set_color(WHITE)
        gpt3_text = Text("GPT-3 (Released 2020)").scale(0.5).next_to(gpt3, RIGHT)
        self.add(gpt3)
        self.play(Write(gpt3_text))
        self.wait(1)
        self.remove(attention_paper)
        self.play(VGroup(gpt1, gpt1_text, gpt2, gpt2_text, gpt3, gpt3_text).animate.to_edge(LEFT))
        chatgpt = SVGMobject("./img/openai.svg").scale(1).set_color(GREEN)
        chatgpt_text = Text("ChatGPT").scale(1).next_to(chatgpt, DOWN)
        self.add(chatgpt)
        self.play(Write(chatgpt_text))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        goal1 = Rectangle(height=5, width=6.5).to_edge(LEFT)
        self.play(Create(goal1))
        self.wait(1)
        goal2 = Rectangle(height=5, width=6.5).to_edge(RIGHT)
        self.play(Create(goal2))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)