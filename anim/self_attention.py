import numpy as np
from manim import *
import pickle

from utils import remove_invisible_chars

class Dataset(Scene):
    def construct(self):
        # Setting background colour
        self.camera.background_color = "#282a36"
        # Self Attention box
        title = Text("Self Attention")
        box = SurroundingRectangle(title, color=WHITE, buff=MED_LARGE_BUFF)
        self.play(Write(title))
        self.wait(1)
        self.play(Create(box))
        self.wait(1)
        # Demo Input and output sequences
        in_seq = Matrix([[1, 2, 3, 4, 5]]).scale(0.6).next_to(box, LEFT)
        out_seq = Matrix([[5, 4, 3, 2, 1]]).scale(0.6).next_to(box, RIGHT)
        self.play(Write(in_seq))
        self.wait(1)
        self.play(FadeOut(in_seq, target_position=title))
        self.play(FadeIn(out_seq, target_position=title))
        self.wait(1)
        self.play(FadeOut(title, out_seq))
        self.play(box.animate.scale(8), run_time=2)
        self.wait(1)

        # IMDB dataset
        title = Text("IMDB Dataset").to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        with open ('review1.pkl', 'rb') as fp:
            review1 = pickle.load(fp)
        review1 = Text(review1).scale(0.3).next_to(title, DOWN)
        label = Text("Label: 1").scale(0.5).next_to(review1, DOWN)
        self.play(Write(review1))
        self.wait(1)
        self.play(Write(label))
        self.wait(1)
        comment = Text("There are 20000 more reviews just like this!").scale(0.5).next_to(label, DOWN).shift(3*DOWN)
        self.play(Write(comment))
        self.wait(1)
        self.play(FadeOut(label))
        with open ('review2.pkl', 'rb') as fp:
            review2 = pickle.load(fp)
        review2 = Text(review2).scale(0.3).next_to(review1, DOWN)
        self.play(Write(review2))
        with open ('review3.pkl', 'rb') as fp:
            review3 = pickle.load(fp)
        review3 = Text(review3).scale(0.3).next_to(review2, DOWN)
        self.play(Write(review3))
        vdots = Tex(r"$\vdots$").scale(0.5).next_to(review3, DOWN)
        with open ('review4.pkl', 'rb') as fp:
            review4 = pickle.load(fp)
        review4 = Text(review4).scale(0.3).next_to(vdots, DOWN)
        self.play(ReplacementTransform(comment, VGroup(vdots, review4)))
        self.wait(1)

        # Tokenization and Dictionary building
        with open ('vocab2.pkl', 'rb') as fp:
            vocab2 = pickle.load(fp)
        vocab2 = Text(vocab2).scale(0.3).to_edge(DOWN)
        vdots_ = Tex(r"\vdots").scale(0.5).next_to(vocab2, UP)
        with open ('vocab1.pkl', 'rb') as fp:
            vocab1 = pickle.load(fp)
        vocab1 = Text(vocab1).scale(0.3).next_to(vdots_, UP)
        vocab_box = SurroundingRectangle(VGroup(vocab1, vdots_, vocab2), color=WHITE, buff=MED_SMALL_BUFF)
        self.play(Create(vocab_box))
        self.wait(1)
        self.play(ReplacementTransform(VGroup(review1, review2, review3, vdots, review4), VGroup(vocab1, vdots_, vocab2)))
        self.wait(1)

        with open ('num_vocab2.pkl', 'rb') as fp:
            num_vocab2 = pickle.load(fp)
        num_vocab2 = Text(num_vocab2).scale(0.3).to_edge(DOWN)
        num_vdots_ = Tex(r"\vdots").scale(0.5).next_to(num_vocab2, UP)
        with open ('num_vocab1.pkl', 'rb') as fp:
            num_vocab1 = pickle.load(fp)
        num_vocab1 = Text(num_vocab1).scale(0.3).next_to(num_vdots_, UP)
        self.wait(1)
        self.play(ReplacementTransform(VGroup(vocab1, vdots_, vocab2), VGroup(num_vocab1, num_vdots_, num_vocab2)))
        self.wait(1)

        with open ('review1.pkl', 'rb') as fp:
            review1 = pickle.load(fp)
        review1 = Text(review1).scale(0.3).next_to(title, DOWN)
        self.play(Write(review1))
        with open ('review2.pkl', 'rb') as fp:
            review2 = pickle.load(fp)
        review2 = Text(review2).scale(0.3).next_to(review1, DOWN)
        self.play(Write(review2))
        with open ('review3.pkl', 'rb') as fp:
            review3 = pickle.load(fp)
        review3 = Text(review3).scale(0.3).next_to(review2, DOWN)
        self.play(Write(review3))
        vdots = Tex(r"$\vdots$").scale(0.5).next_to(review3, DOWN)
        with open ('review4.pkl', 'rb') as fp:
            review4 = pickle.load(fp)
        review4 = Text(review4).scale(0.3).next_to(vdots, DOWN)  
        self.play(Write(vdots), Write(review4))  
        
        with open ('num_review1.pkl', 'rb') as fp:
            num_review1 = pickle.load(fp)
        num_review1 = Text(num_review1).scale(0.3).move_to(review1, DOWN)
        self.play(ReplacementTransform(review1, num_review1))
        with open ('num_review2.pkl', 'rb') as fp:
            num_review2 = pickle.load(fp)
        num_review2 = Text(num_review2).scale(0.3).move_to(review2, DOWN)
        self.play(ReplacementTransform(review2, num_review2))
        with open ('num_review3.pkl', 'rb') as fp:
            num_review3 = pickle.load(fp)
        num_review3 = Text(num_review3).scale(0.3).move_to(review3, DOWN)
        self.play(ReplacementTransform(review3, num_review3))
        with open ('num_review4.pkl', 'rb') as fp:
            num_review4 = pickle.load(fp)
        num_review4 = Text(num_review4).scale(0.3).move_to(review4, DOWN)  
        self.play(ReplacementTransform(review4, num_review4))  
        self.wait(1)

        self.play(FadeOut(vocab_box, num_vocab1, num_vdots_, num_vocab2))
        comment = Text("Different sentences have different lengths...").scale(0.5).next_to(label, DOWN).shift(3*DOWN)
        self.play(Write(comment))
        self.wait(1)
        comment1 = Text("Padding to make them of equal lengths!").scale(0.5).next_to(label, DOWN).shift(3*DOWN)

        with open ('padded_num_review1.pkl', 'rb') as fp:
            padded_num_review1 = pickle.load(fp)
        padded_num_review1 = Text(padded_num_review1).scale(0.3).move_to(review1, DOWN)
        self.play(ReplacementTransform(num_review1, padded_num_review1), ReplacementTransform(comment, comment1))
        with open ('padded_num_review2.pkl', 'rb') as fp:
            padded_num_review2 = pickle.load(fp)
        padded_num_review2 = Text(padded_num_review2).scale(0.3).move_to(review2, DOWN)
        self.play(ReplacementTransform(num_review2, padded_num_review2))
        with open ('padded_num_review3.pkl', 'rb') as fp:
            padded_num_review3 = pickle.load(fp)
        padded_num_review3 = Text(padded_num_review3).scale(0.3).move_to(review3, DOWN)
        self.play(ReplacementTransform(num_review3, padded_num_review3))
        with open ('padded_num_review4.pkl', 'rb') as fp:
            padded_num_review4 = pickle.load(fp)
        padded_num_review4 = Text(padded_num_review4).scale(0.3).move_to(review4, DOWN)  
        self.play(ReplacementTransform(num_review4, padded_num_review4))  
        self.wait(1)

        self.play(FadeOut(comment1, padded_num_review4, padded_num_review3, padded_num_review2, vdots, padded_num_review1[10:]))
        ex_review = VGroup(Text("14"),
                           Text("19"),
                           Text("10"),
                           Text("59"),
                           Text("57")).scale(0.5).arrange(DOWN, aligned_edge=LEFT)
        self.play(ReplacementTransform(padded_num_review1[:10], ex_review))
        self.wait(1)

        emb_token = np.load("ex_review_emb.npy")
        ex_review_emb = Matrix(np.round(emb_token, 2), v_buff=1.5, h_buff=1.5).scale(0.5).next_to(title, DOWN).shift(0.5*DOWN)
        self.play(ReplacementTransform(ex_review, ex_review_emb))
        self.wait(1)

        emb_code = Code(code="""
                            token_embedding = torch.nn.Embedding(embedding_dim=7, num_embeddings=len(vocab))
                            emb_token = token_embedding(ex_review)
                            """, language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.5).next_to(ex_review_emb, DOWN).shift(DOWN)
        emb_code.code = remove_invisible_chars(emb_code.code)
        self.play(Create(emb_code))
        self.wait(1)
        box1 = SurroundingRectangle(emb_code.code[0][35:50], buff=0.05)
        t1 = Text("Vector Length").scale(0.3).next_to(box1, UP)
        box2 = SurroundingRectangle(emb_code.code[0][51:-1], buff=0.05)
        t2 = Text("Vocab Size").scale(0.3).next_to(box2, UP)
        self.play(Create(box1), Write(t1))
        self.wait(1)
        self.play(Create(box2), Write(t2))
        self.wait(1)
        self.play(FadeOut(emb_code, box1, box2, t1, t2))

        mat_dim = Tex(r"1", r"$\times$", r"5", r"$\times$", r"7").scale(2).next_to(ex_review_emb, DOWN).shift(0.5*DOWN)
        t1 = Text("Total Reviews").scale(0.3).next_to(mat_dim[0], DOWN)
        t2 = Text("Total Words").scale(0.3).next_to(mat_dim[2], UP)
        t3 = Text("Embedding Size").scale(0.3).next_to(mat_dim[-1], DOWN)
        self.play(Write(mat_dim))
        self.wait(1)
        self.play(Write(t1))
        self.wait(1)
        self.play(Write(t2))
        self.wait(1)
        self.play(Write(t3))
        self.wait(1)
        self.play(FadeOut(title))
        self.wait(1)

        return super().construct()